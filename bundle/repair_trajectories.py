"""Replay recorded tool calls to fill in tool results lost when trajectories were recorded.

A recording bug turned sql_query results (JSON lists of rows) into empty strings. The tools
are deterministic over the seeded SQLite database, so each lost result is recomputed exactly
from the tool call that produced it. Runs as a Code Bundle; trace data stays in Snowflake.

Args: --run-tags TAG,TAG  --suite hard2
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--run-tags", required=True)
ap.add_argument("--suite", default="hard2")
args, _unknown = ap.parse_known_args()

from snowflake.snowpark import Session  # noqa: E402

sp = Session.builder.getOrCreate()
sp.use_database("CT_QUICKSTART_DB")
sp.use_schema("AGT")


def log(msg: str):
    sp.sql("INSERT INTO CT_QUICKSTART_DB.AGT.RUN_LOG SELECT 'repair', CURRENT_TIMESTAMP(), ?", params=[msg[:15000]]).collect()


try:
    for cand in (os.getcwd(), *[r for r, _, f in os.walk(os.getcwd()) if "agent_lib.py" in f][:1]):
        if os.path.exists(os.path.join(cand, "agent_lib.py")):
            sys.path.insert(0, cand)
            break
    import agent_lib as L

    db, _tasks, _prompt = L.suite(args.suite)
    tags = args.run_tags.split(",")
    rows = sp.sql(f"SELECT run_tag, record_id, chat_messages FROM TRAJECTORIES WHERE run_tag IN ({','.join('?' * len(tags))}) "
                  "AND chat_messages IS NOT NULL", params=tags).collect()
    t0, fixed_msgs, mismatched, out = time.time(), 0, 0, []
    for r in rows:
        msgs = json.loads(r["CHAT_MESSAGES"])
        pending = []  # tool calls waiting for their results, in order
        for m in msgs:
            if m["role"] == "assistant":
                pending = list(m.get("tool_calls") or [])
            elif m["role"] == "tool":
                if not pending:
                    mismatched += 1
                    continue
                call = pending.pop(0)["function"]
                a = call.get("arguments") or {}
                if isinstance(a, str):
                    a = json.loads(a)
                result = (L.run_sql(db, a.get("query", "")) if call["name"] == "sql_query"
                          else L.run_calculator(a.get("expression", "")))
                if m.get("content", "") != result:
                    if m.get("content"):
                        mismatched += 1  # a recorded non-empty result differs; keep the replayed one
                    m["content"] = result
                    fixed_msgs += 1
        out.append([r["RUN_TAG"], r["RECORD_ID"], json.dumps(msgs)])
    log(f"replayed {len(rows)} trajectories in {time.time() - t0:.0f}s: {fixed_msgs} tool messages filled, "
        f"{mismatched} non-empty results that differed or had no matching call")
    sp.create_dataframe(out, schema=["RUN_TAG", "RECORD_ID", "MSGS"]).write.mode("overwrite").save_as_table("TRAJ_REPAIR")
    sp.sql("UPDATE TRAJECTORIES t SET chat_messages = PARSE_JSON(r.msgs) FROM TRAJ_REPAIR r "
           "WHERE t.run_tag = r.run_tag AND t.record_id = r.record_id").collect()
    sp.sql("DROP TABLE TRAJ_REPAIR").collect()
    log("DONE repair " + args.run_tags)
except Exception:
    import traceback

    log("FAILED repair " + traceback.format_exc())
    raise
