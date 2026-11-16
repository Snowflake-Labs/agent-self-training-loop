"""Post-train Qwen3-8B (LoRA) with Cortex Training on the best Opus trajectories.

Runs as a Code Bundle on CT_QUICKSTART_CPU. Reads CT_QUICKSTART_DB.AGT.TRAINING_SET
(built in SQL from AGENT_TRACE_TABLE); the data goes only to this account's Cortex
Training API. Writes the job/checkpoint and per-step losses to AGT tables.

Args: --run-tag TAG --source-run-tag TAG --epochs N --batch-size N --lr F
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from importlib import metadata
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run-tag", required=True)
ap.add_argument("--source-run-tag", required=True)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max-seq-len", type=int, default=8192)
ap.add_argument("--suite", default="hard", choices=["basic", "hard", "hard2"])
ap.add_argument("--view", default="TRAINING_SET", choices=["TRAINING_SET", "TRAINING_SET_JUDGED"])
ap.add_argument("--dry-run", action="store_true", help="convert + tokenize only; no GPUs")
args, _ = ap.parse_known_args()

from snowflake.snowpark import Session  # noqa: E402

sp = Session.builder.getOrCreate()
sp.use_database("CT_QUICKSTART_DB")
sp.use_schema("AGT")
sp.sql("CREATE TABLE IF NOT EXISTS RUN_LOG(run_tag STRING, logged_at TIMESTAMP_LTZ, msg STRING)").collect()


def log(msg):
    sp.sql("INSERT INTO RUN_LOG SELECT ?, CURRENT_TIMESTAMP(), ?", params=[args.run_tag, str(msg)[:15000]]).collect()


# SNOWFLAKE_HOST if set, else the hyphenated <org>-<account> host of the session's account.
_org, _account = sp.sql("SELECT CURRENT_ORGANIZATION_NAME(), CURRENT_ACCOUNT_NAME()").collect()[0]
HOST = os.environ.get("SNOWFLAKE_HOST") or f"{_org}-{_account}".lower().replace("_", "-") + ".snowflakecomputing.com"
MODEL_NAME = "Qwen/Qwen3-8B"
TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SDK_REVISION = "b4c26d57bcd741dce4cd63a6ed8f51cff6ba8829"
SDK_SHA256 = "a7860f487ad542204cc0f2c39137ac1be0ef97d9845942d11021e991d839b4df"
TRAINING_GPUS, SAMPLING_GPUS, HARDWARE = 4, 4, "H200"

# Must match the tool schemas the agent exposes (agent_lib.build_agent) and the eval prompt.
TOOLS = [
    {"type": "function", "function": {
        "name": "sql_query", "description": "Run a read-only SQLite query and return up to 50 rows as JSON.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression with + - * / ( ) and decimals.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]


def semconv_to_chat(system_instructions, input_messages, output_messages, student_system):
    """Gateway GenAI-semconv messages -> HF chat messages with native tool calls."""
    msgs = [{"role": "system", "content": student_system}]
    for m in list(input_messages) + list(output_messages):
        parts = m.get("parts", [])
        if m["role"] == "assistant":
            text = "".join(p.get("content", "") for p in parts if p.get("type") == "text").strip()
            calls = [{"type": "function", "function": {"name": p["name"], "arguments": p.get("arguments") or {}}}
                     for p in parts if p.get("type") == "tool_call"]
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            msgs.append(msg)
        else:
            responses = [p for p in parts if p.get("type") == "tool_call_response"]
            if responses:
                msgs.extend({"role": "tool", "content": str(p.get("response", ""))} for p in responses)
            text = "".join(p.get("content", "") for p in parts if p.get("type") == "text").strip()
            if text:
                msgs.append({"role": "user", "content": text})
    if msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls"):
        raise ValueError("trajectory does not end with a final assistant answer")
    return msgs


try:
    t0 = time.time()
    constraints = [f"{p}=={metadata.version(p)}" for p in
                   ("torch", "snowflake-snowpark-python", "snowflake-connector-python", "numpy")]
    with tempfile.TemporaryDirectory() as d:
        cpath = Path(d) / "c.txt"
        cpath.write_text("\n".join(constraints) + "\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check",
                        "--constraint", str(cpath),
                        f"cortex-training @ https://github.com/snowflakedb/cortex-training/archive/{SDK_REVISION}.zip#sha256={SDK_SHA256}",
                        "transformers==4.57.6", "jinja2>=3.1"], check=True, capture_output=True, text=True)
    log(f"pip ok {time.time() - t0:.0f}s")

    import torch
    from cortex_training import CortexTrainingClient, wire
    from snowflake.ml.experiment import ExperimentTracking
    from snowflake.snowpark.secrets import get_generic_secret_string
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path.cwd()))
    import agent_lib as L

    STUDENT_SYSTEM = L.SYSTEM_PROMPTS[args.suite] + " /no_think"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=TOKENIZER_REVISION)
    end_id = tok.convert_tokens_to_ids("<|im_end|>")

    def render(msgs, gen_prompt=False):
        return tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False,
                                       add_generation_prompt=gen_prompt, enable_thinking=False)

    def encode(msgs):
        """One example per assistant turn, each with that turn as the only target.

        Qwen3's template drops the empty <think></think> block from earlier assistant
        turns but adds it to the generation prompt, so a single whole-trajectory render
        does not match what the model sees at each step. Rendering turn i as the
        continuation of the generation prompt for msgs[:i] matches inference exactly.
        """
        out = []
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            prefix = render(msgs[:i], gen_prompt=True)
            through = render(msgs[: i + 1])
            if not through.startswith(prefix):
                raise ValueError("chat template is not prefix-stable for this turn")
            ids = tok(through, add_special_tokens=False)["input_ids"]
            a = len(tok(prefix, add_special_tokens=False)["input_ids"])
            b = len(ids)
            # Target ends at this turn's <|im_end|>; the template's trailing newline is not a target.
            while b > a and ids[b - 1] != end_id:
                b -= 1
            if b <= a:
                raise ValueError("empty assistant turn")
            weights = [0.0] * len(ids)
            for j in range(a, b):
                weights[j] = 1.0
            out.append((ids, weights[1:] + [0.0]))  # weight for predicting token j+1 sits at position j
        if not out:
            raise ValueError("no assistant turns")
        return out

    def to_chat(r):
        if r["CHAT_MESSAGES"]:  # recorded trajectory span (deployment calls that bypass the gateway)
            return [{"role": "system", "content": STUDENT_SYSTEM}] + json.loads(r["CHAT_MESSAGES"])
        return semconv_to_chat(json.loads(r["SYSTEM_INSTRUCTIONS"] or "[]"), json.loads(r["INPUT_MESSAGES"]),
                               json.loads(r["OUTPUT_MESSAGES"]), STUDENT_SYSTEM)

    sources = args.source_run_tag.split(",")
    rows = sp.sql("SELECT run_tag, task_id, record_id, system_instructions, input_messages, output_messages, chat_messages "
                  f"FROM {args.view} WHERE run_tag IN ({','.join('?' * len(sources))}) ORDER BY run_tag, task_id, rank_in_task",
                  params=sources).collect()
    examples, skipped, lengths = [], {}, []
    for r in rows:
        try:
            msgs = to_chat(r)
            turns = encode(msgs)
            if max(len(ids) for ids, _ in turns) > args.max_seq_len:
                raise ValueError("longer than max_seq_len")
            for ids, w in turns:
                examples.append((r["TASK_ID"], ids, w))
                lengths.append(len(ids))
        except Exception as e:
            skipped[str(e)[:80]] = skipped.get(str(e)[:80], 0) + 1
    lengths.sort()
    stats = {"rows": len(rows), "by_source": {s: sum(1 for r in rows if r["RUN_TAG"] == s) for s in sources},
             "trajectories": len(rows) - sum(skipped.values()), "turn_examples": len(examples),
             "skipped": skipped,
             "tasks": len({e[0] for e in examples}),
             "len_p50": lengths[len(lengths) // 2] if lengths else None, "len_max": lengths[-1] if lengths else None,
             "target_tokens": int(sum(sum(e[2]) for e in examples))}
    log(f"data {json.dumps(stats)}")
    if examples:
        sample = to_chat(rows[-1])
        for k, (ids, w) in enumerate(encode(sample)):
            log(f"sample turn {k} targets: " + tok.decode([t for t, wt in zip(ids[1:], w[:-1]) if wt > 0])[:1500])
    if args.dry_run or not examples:
        log("DONE dry-run")
        raise SystemExit(0)

    max_seq = min(args.max_seq_len, 1 << math.ceil(math.log2(max(lengths) + 1)))
    steps = max(1, math.ceil(args.epochs * len(examples) / args.batch_size))

    try:
        pat = get_generic_secret_string("ct_quickstart_db/agt/agt_pat")
    except Exception:
        pat = get_generic_secret_string("ct_quickstart_db.agt.agt_pat")
    client = CortexTrainingClient.from_pat(host=HOST, pat=pat, database="CT_QUICKSTART_DB", schema="LAB",
                                           poll_timeout=1800, poll_interval=1.0,
                                           request_timeout=(10.0, 300.0), max_retries=0)
    cap = client.get_capacity(hardware=HARDWARE)
    log(f"capacity {cap} max_seq={max_seq} steps={steps}")
    if cap["available_gpus"] < TRAINING_GPUS + SAMPLING_GPUS:
        raise RuntimeError("not enough free GPUs; drop deployments first")

    lora = {"peft_type": "Lora", "r": 32, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    body = {"hardware": HARDWARE, "idle_timeout_seconds": 1800, "sub_job_configs": [
        {"job_type": "sampling", "model_name": MODEL_NAME, "dtype": "bfloat16", "seed": 42,
         "inference_config": {"max_seq_len": max_seq, "n_gpus": SAMPLING_GPUS,
                              "vllm_config": {"tensor_parallel_size": SAMPLING_GPUS, "gpu_memory_utilization": 0.8},
                              "peft_config": deepcopy(lora)}},
        {"job_type": "training", "model_name": MODEL_NAME, "dtype": "bfloat16", "seed": 42,
         "training_config": {"n_gpus": TRAINING_GPUS, "max_seq_len": max_seq, "train_batch_size": args.batch_size,
                             "gradient_clipping": 1.0, "model_provider": "huggingface",
                             "attn_implementation": "flash_attention_3",
                             "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": 0.0,
                                           "betas": [0.9, 0.95], "eps": 1e-8},
                             "ds_config": {"zero_optimization": {"stage": 2, "reduce_scatter": True},
                                           "bf16": {"enabled": True}},
                             "peft_config": deepcopy(lora)}},
    ]}
    CE = {"loss_fn": "causal_cross_entropy", "post": ["compute_logprobs"], "config": {}}
    TERMINAL = {"cancelled", "failed", "done", "terminated"}

    def status(job_id):
        return str(client.get_job(job_id).get("status", "")).lower().removeprefix("job_state_")

    def to_batch(batch):
        width = max(len(ids) for _, ids, _ in batch)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for _, ids, _ in batch], dtype=torch.long)
        kwargs = {"input_ids": input_ids,
                  "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for _, ids, _ in batch]),
                  "position_ids": torch.arange(width).expand(len(batch), -1).contiguous(), "use_cache": False}
        ctx = {"input_ids": input_ids,
               "loss_mask": torch.tensor([w + [0.0] * (width - len(w)) for _, _, w in batch], dtype=torch.float32)}
        return kwargs, ctx

    job_id, tracker, run_name = None, None, None
    try:
        job_id = client.create_job_from_body(deepcopy(body))["job_id"]
        sp.sql("CREATE TABLE IF NOT EXISTS TRAINING_RUNS(run_tag STRING, source_run_tag STRING, job_id STRING, "
               "checkpoint_id STRING, examples INT, steps INT, max_seq_len INT, created_at TIMESTAMP_LTZ)").collect()
        sp.sql("INSERT INTO TRAINING_RUNS SELECT ?, ?, ?, NULL, ?, ?, ?, CURRENT_TIMESTAMP()",
               params=[args.run_tag, args.source_run_tag, job_id, len(examples), steps, max_seq]).collect()
        log(f"job {job_id} created; waiting for GPUs")
        client.wait_for_job(job_id)
        try:
            info = client.get_experiment_run(job_id)
            run_name = info["experiment_run_name"]
            tracker = ExperimentTracking(session=sp)
            tracker.set_experiment(info["experiment_name"].split(".")[-1].strip('"'))
            tracker.start_run(run_name)
            tracker.log_params({"model": MODEL_NAME, "source_run_tag": args.source_run_tag, "examples": len(examples),
                                "steps": steps, "batch_size": args.batch_size, "lr": args.lr, "max_seq_len": max_seq})
        except Exception as e:
            tracker = None
            log(f"experiment tracking unavailable: {e}")
        log(f"job running after {time.time() - t0:.0f}s")
        rng = random.Random(0)
        order = []
        losses = []
        for step in range(1, steps + 1):
            if len(order) < args.batch_size:
                order += rng.sample(examples, len(examples))
            batch, order = order[: args.batch_size], order[args.batch_size:]
            kwargs, ctx = to_batch(batch)
            payload = wire.dumps({"args": (), "kwargs": kwargs, "context": ctx, "processing": CE},
                                 metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}})
            loss = float(client.poll_request(job_id, client.forward_backward(job_id, payload))["avg_loss"])
            if not math.isfinite(loss):
                raise RuntimeError("non-finite loss")
            client.poll_request(job_id, client.step(job_id, learning_rate=args.lr))
            losses.append(loss)
            if tracker:
                tracker.log_metrics({"train_loss": loss}, step=step)
            if step % 5 == 0 or step == steps or step == 1:
                log(f"step {step}/{steps} loss {loss:.4f}")

        previous = {c["checkpoint_id"] for c in client.list_checkpoints(job_id)}
        saved = client.poll_request(job_id, client.save(job_id, checkpoint_type="weights-only"))
        checkpoint_id = saved.get("checkpoint_id")
        deadline = time.monotonic() + 300
        while checkpoint_id is None:
            new = [c["checkpoint_id"] for c in client.list_checkpoints(job_id)
                   if c.get("checkpoint_type") == "weights-only" and c["checkpoint_id"] not in previous]
            checkpoint_id = new[0] if len(new) == 1 else None
            if checkpoint_id is None:
                if time.monotonic() > deadline:
                    raise TimeoutError("checkpoint not identifiable")
                time.sleep(5)
        sp.sql("UPDATE TRAINING_RUNS SET checkpoint_id = ? WHERE job_id = ?", params=[checkpoint_id, job_id]).collect()
        log(f"checkpoint {checkpoint_id} losses first={losses[0]:.4f} last5={sum(losses[-5:]) / min(5, len(losses)):.4f}")
        ok = True
    except BaseException:
        ok = False
        raise
    finally:
        if job_id:
            try:
                if status(job_id) not in TERMINAL:
                    client.cancel_job(job_id)
                deadline = time.monotonic() + 600
                while status(job_id) not in TERMINAL and time.monotonic() < deadline:
                    time.sleep(5)
                log(f"job {job_id} released status={status(job_id)}")
            finally:
                if tracker and run_name:
                    try:
                        tracker.end_run(run_name, status="FINISHED" if ok else "FAILED")
                    except Exception:
                        pass
    log(f"DONE {time.time() - t0:.0f}s")
except SystemExit:
    pass
except BaseException:
    log("FAILED " + traceback.format_exc())
    raise
