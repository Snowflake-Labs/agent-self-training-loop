"""In-Snowflake batch: LangGraph agent rollouts through AI Gateway, TruLens app + EVAL
spans teed to the gateway OTLP endpoint, feedbacks judged by Cortex.

Runs as a Code Bundle on CT_QUICKSTART_CPU. Trace content goes only to the gateway's
event table; the local TruLens SQLite file lives in the container and is deleted at
the end. Progress is written to CT_QUICKSTART_DB.AGT.RUN_LOG (print output is not
captured for bundles).

Args: --run-tag TAG --rollouts N --limit N --workers N
"""
import argparse
import os
import subprocess
import sys
import time
import traceback

PINS = [
    "trulens-core==2.16.0", "trulens-feedback==2.16.0", "trulens-apps-langgraph==2.16.0",
    "trulens-apps-langchain==2.16.0", "trulens-providers-cortex==2.16.0", "trulens-otel-semconv==2.16.0",
    "langgraph==1.2.19", "langgraph-prebuilt==1.1.1", "langchain-core==1.7.5", "langchain-openai==1.7.3",
    "openai==3.45.0", "opentelemetry-sdk==1.45.2", "opentelemetry-exporter-otlp-proto-http==1.45.2",
    "opentelemetry-instrumentation-httpx==0.66b2", "langchain-anthropic==1.7.9", "anthropic==2.4.0",
]

ap = argparse.ArgumentParser()
ap.add_argument("--run-tag", required=True)
ap.add_argument("--rollouts", type=int, default=3)
ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--splits", default="train,heldout")
ap.add_argument("--suite", default="basic", choices=["basic", "hard", "hard2"])
ap.add_argument("--kinds", default="", help="comma-separated task-id prefixes to keep, e.g. n (new kinds only)")
ap.add_argument("--model", default=None, help="defaults to agent_lib.TEACHER_MODEL")
ap.add_argument("--base-url", default=None, help="OpenAI-compatible base URL, e.g. a Cortex Training deployment")
ap.add_argument("--sampler", default=None,
                help="JOB_ID:CHECKPOINT_ID - serve that checkpoint from a Cortex Training sampling job "
                     "(pre-rendered prompts) instead of a deployment's chat endpoint")
ap.add_argument("--sampler-gpus", type=int, default=4)
ap.add_argument("--rerun-errors-of", default=None,
                help="RUN_TAG - rerun only the (task, rollout) pairs that errored in that run, per ROLLOUT_INDEX")
args, _unknown = ap.parse_known_args()

HOST = "sfdevrel-enterprise.snowflakecomputing.com"
SDK_REVISION = "b4c26d57bcd741dce4cd63a6ed8f51cff6ba8829"  # same pin as train_from_traces.py
SDK_SHA256 = "a7860f487ad542204cc0f2c39137ac1be0ef97d9845942d11021e991d839b4df"

from snowflake.snowpark import Session  # noqa: E402

sp = Session.builder.getOrCreate()
sp.use_database("CT_QUICKSTART_DB")
sp.use_schema("AGT")
sp.sql("CREATE TABLE IF NOT EXISTS CT_QUICKSTART_DB.AGT.RUN_LOG(run_tag STRING, logged_at TIMESTAMP_LTZ, msg STRING)").collect()


def log(msg: str):
    sp.sql("INSERT INTO CT_QUICKSTART_DB.AGT.RUN_LOG SELECT ?, CURRENT_TIMESTAMP(), ?", params=[args.run_tag, msg[:15000]]).collect()


try:
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *PINS],
                   check=True, capture_output=True, text=True)
    if args.sampler:
        from importlib import metadata as _md
        pinned = [f"{p}=={_md.version(p)}" for p in ("torch", "numpy")]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *pinned,
                        f"cortex-training @ https://github.com/snowflakedb/cortex-training/archive/{SDK_REVISION}.zip#sha256={SDK_SHA256}",
                        "transformers==4.57.6", "jinja2>=3.1"], check=True, capture_output=True, text=True)
    log(f"pip install ok {time.time() - t0:.0f}s")

    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for cand in (here, os.getcwd(), *[os.path.join(r, "") for r, _, f in os.walk(os.getcwd()) if "agent_lib.py" in f][:1]):
        if os.path.exists(os.path.join(cand, "agent_lib.py")):
            sys.path.insert(0, cand)
            break
    else:
        raise FileNotFoundError(f"agent_lib.py not found from cwd={os.getcwd()}")
    import tempfile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import agent_lib as L
    from snowflake.snowpark.secrets import get_generic_secret_string
    from trulens.apps.langgraph import TruGraph
    from trulens.core import TruSession
    from trulens.core.database.connector.default import DefaultDBConnector

    try:
        pat = get_generic_secret_string("ct_quickstart_db/agt/agt_pat")
    except Exception:
        pat = get_generic_secret_string("ct_quickstart_db.agt.agt_pat")

    tmp = tempfile.mkdtemp(prefix="trulens-")
    connector = DefaultDBConnector(database_url=f"sqlite:///{tmp}/t.sqlite")
    tee = L.make_tee_exporter(connector, pat)
    session = TruSession(connector=connector, span_exporter=tee)

    db, tasks, system_prompt = L.suite(args.suite)
    sp.create_dataframe([[t["task_id"], t["kind"], t["question"], str(t["answer"]), t["split"]] for t in tasks],
                        schema=["TASK_ID", "KIND", "QUESTION", "ANSWER", "SPLIT"]) \
      .write.mode("overwrite").save_as_table(f"CT_QUICKSTART_DB.AGT.TASKS_{args.suite.upper()}")
    splits = set(args.splits.split(","))
    chosen = [t for t in tasks if t["split"] in splits]
    if args.kinds:
        chosen = [t for t in chosen if t["task_id"].startswith(tuple(args.kinds.split(",")))]
    if args.limit:
        chosen = chosen[: args.limit]
    log(f"tasks total={len(tasks)} chosen={len(chosen)} rollouts={args.rollouts} workers={args.workers}")

    feedbacks = L.make_feedbacks(sp, {t["question"]: t for t in tasks})
    # One TruGraph wrapper; each rollout builds its own graph so the conversation-id header is per rollout.
    proto = L.build_agent(db, pat, "proto")
    tru = TruGraph(proto, app_name="retail_agent", app_version=args.run_tag, feedbacks=feedbacks)
    # TruLens allows one active recording per app instance, so each worker thread gets its own wrapper.
    import threading
    local = threading.local()
    lock = threading.Lock()

    def worker_tru():
        if not hasattr(local, "tru"):
            with lock:
                local.tru = TruGraph(L.build_agent(db, pat, "proto"), app_name="retail_agent",
                                     app_version=args.run_tag, feedbacks=feedbacks)
        return local.tru

    import json as _json
    record_trajectory = L.make_trajectory_recorder(session, "retail_agent", args.run_tag)

    sampler = None
    if args.sampler:
        from cortex_training import CortexTrainingClient

        src_job, src_ckpt = args.sampler.split(":")
        ct = CortexTrainingClient.from_pat(host=HOST, pat=pat, database="CT_QUICKSTART_DB", schema="LAB",
                                           poll_timeout=1800, poll_interval=0.5)
        sampler = L.Sampler.start(ct, src_job, src_ckpt, n_gpus=args.sampler_gpus)
        log(f"sampler job {sampler.job_id} from {args.sampler}; waiting for GPUs")
        ct.wait_for_job(sampler.job_id)
        log(f"sampler running {time.time() - t0:.0f}s")

    def run_one(task, k):
        conv = f"{args.run_tag}:{task['task_id']}:{k}"
        graph = L.build_agent(db, pat, conv, model=args.model or L.TEACHER_MODEL, base_url=args.base_url,
                              system_prompt=system_prompt, sampler=sampler)
        meta = {"task_id": task["task_id"], "split": task["split"], "conversation_id": conv,
                "run_tag": args.run_tag, "rollout": k}
        t = time.time()
        res = None
        with worker_tru() as rec:
            try:
                res = graph.invoke({"messages": [{"role": "user", "content": task["question"]}]},
                                   config={"metadata": meta, "recursion_limit": 24})
                text, err = L.final_text(res), None
            except Exception as e:  # keep failed rollouts; they are filtered out later
                text, err = "", f"{type(e).__name__}: {e}"[:500]
        if res is not None:  # outside the recording, so it is not a TruLens record
            record_trajectory(conv, _json.dumps(L.trajectory_messages(res)))
        try:
            rid = rec.get().record_id
        except Exception:
            rid = conv  # rows join on conversation_id anyway
        return [task["task_id"], k, task["split"], conv, rid, L.is_correct(text, task["answer"]), err, round(time.time() - t, 2)]

    rows, done = [], 0
    jobs = [(t, k) for t in chosen for k in range(args.rollouts)]
    if args.rerun_errors_of:
        errored = {(r[0], r[1]) for r in sp.sql(
            "SELECT TASK_ID, ROLLOUT FROM CT_QUICKSTART_DB.AGT.ROLLOUT_INDEX "
            "WHERE STARTSWITH(CONVERSATION_ID, ?) AND ERROR IS NOT NULL",
            params=[args.rerun_errors_of + ":"]).collect()}
        jobs = [(t, k) for t, k in jobs if (t["task_id"], k) in errored]
        log(f"rerunning {len(jobs)} errored rollouts of {args.rerun_errors_of}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_one, t, k) for t, k in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 25 == 0 or done == len(jobs):
                log(f"rollouts {done}/{len(jobs)} correct={sum(r[5] for r in rows)} errors={sum(1 for r in rows if r[6])}")
    if sampler is not None:  # rollouts are done; release the GPUs before judging
        sampler.client.cancel_job(sampler.job_id)
        log(f"sampler job {sampler.job_id} cancelled")
        sampler = None

    sp.create_dataframe(rows, schema=["TASK_ID", "ROLLOUT", "SPLIT", "CONVERSATION_ID", "RECORD_ID", "CORRECT", "ERROR", "SECONDS"]) \
      .write.mode("append").save_as_table("CT_QUICKSTART_DB.AGT.ROLLOUT_INDEX")

    session.force_flush()
    time.sleep(5)
    events = connector.get_events(app_name="retail_agent", app_version=args.run_tag)
    log(f"local spans={len(events)} computing feedbacks")
    tru.compute_feedbacks(events=events, raise_error_on_no_feedbacks_computed=False)
    session.force_flush()
    time.sleep(10)
    session._experimental_tracer_provider.shutdown()
    log(f"DONE remote_spans={tee.remote_spans} remote_failures={tee.remote_failures} elapsed={time.time() - t0:.0f}s")
    subprocess.run(["rm", "-rf", tmp], check=False)
except Exception:
    log("FAILED " + traceback.format_exc())
    raise
finally:
    if globals().get("sampler") is not None:
        try:
            sampler.client.cancel_job(sampler.job_id)
            log(f"sampler job {sampler.job_id} cancelled")
        except Exception as e:
            log(f"sampler cancel failed for {sampler.job_id}: {e}")
