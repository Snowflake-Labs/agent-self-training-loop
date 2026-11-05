# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "trulens-core", "trulens-feedback", "trulens-apps-langgraph", "trulens-providers-cortex",
#   "langgraph", "langchain-openai", "snowflake-snowpark-python",
#   "opentelemetry-exporter-otlp-proto-http", "opentelemetry-instrumentation-httpx",
# ]
# ///
"""One-task smoke test: LangGraph agent through the gateway, TruLens app + EVAL spans
teed to the gateway OTLP endpoint. Local TruLens DB is a temp file deleted at exit."""
import importlib.metadata as md
import json
import os
import sys
import tempfile
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent_lib as L  # noqa: E402

PAT = open(os.path.expanduser("~/.snowflake/agt_lab/pat")).read().strip()
RUN_TAG = "smoke-" + time.strftime("%Y%m%dT%H%M%S")

from snowflake.snowpark import Session  # noqa: E402
from trulens.core import TruSession  # noqa: E402
from trulens.core.database.connector.default import DefaultDBConnector  # noqa: E402
from trulens.apps.langgraph import TruGraph  # noqa: E402

tmp = tempfile.mkdtemp(prefix="trulens-smoke-")
connector = DefaultDBConnector(database_url=f"sqlite:///{tmp}/t.sqlite")
tee = L.make_tee_exporter(connector, PAT)
session = TruSession(connector=connector, span_exporter=tee)
L.instrument_httpx_propagation(session)

sp = Session.builder.config("connection_name", "DEVREL_ENTERPRISE").config("role", "CT_QUICKSTART_USER").create()

db = L.build_db()
tasks = L.make_tasks(db)
task = next(t for t in tasks if t["kind"] == "rev_region_cat")
conv = f"{RUN_TAG}:{task['task_id']}:0"
graph = L.build_agent(db, PAT, conv)
tru = TruGraph(graph, app_name="retail_agent", app_version=RUN_TAG,
               feedbacks=L.make_feedbacks(sp, {t["question"]: t for t in tasks}))

with tru as rec:
    meta = {"task_id": task["task_id"], "split": task["split"], "conversation_id": conv, "run_tag": RUN_TAG}
    result = graph.invoke({"messages": [{"role": "user", "content": task["question"]}]},
                          config={"metadata": meta, "recursion_limit": 24})
answer = L.final_text(result)
print("question:", task["question"]); print("expected:", task["answer"]); print("answer tail:", answer[-160:])
print("programmatic correct:", L.is_correct(answer, task["answer"]))

session.force_flush()
time.sleep(2)
events = connector.get_events(app_name="retail_agent", app_version=RUN_TAG)
print("local spans:", len(events))
for _, e in events.iterrows():
    ra = e["record_attributes"]
    print("  ", e["record"].get("name"), "|", ra.get("ai.observability.span_type"),
          "| trace", (e["trace"] or {}).get("trace_id"))
root = [e for _, e in events.iterrows() if e["record_attributes"].get("ai.observability.span_type") == "record_root"]
if root:
    ra = root[0]["record_attributes"]
    print("root input:", str(ra.get("ai.observability.record_root.input"))[:200])
    print("root config kwarg:", str(ra.get("ai.observability.call.kwargs.config"))[:300])
    print("root output:", str(ra.get("ai.observability.record_root.output"))[:200])
tru.compute_feedbacks(events=events)
session.force_flush()
time.sleep(5)
session.force_flush()
session._experimental_tracer_provider.shutdown()
print("remote spans exported:", tee.remote_spans, "remote export failures:", tee.remote_failures)
print(json.dumps({"run_tag": RUN_TAG, "conversation_id": conv,
                  "versions": {p: md.version(p) for p in ["trulens-core", "trulens-apps-langgraph", "trulens-providers-cortex", "langgraph", "langchain-openai"]}}))
os.system(f"rm -rf {tmp}")
