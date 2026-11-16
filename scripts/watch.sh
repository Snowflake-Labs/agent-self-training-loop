#!/bin/zsh
# Usage: scripts/watch.sh <query_id> <run_tag>
uvx --from snowflake-cli snow bundle status "$1" -c "$SNOWFLAKE_CONNECTION_NAME" --role CT_QUICKSTART_USER 2>&1 | tail -1
uvx --from snowflake-cli snow sql -c "$SNOWFLAKE_CONNECTION_NAME" --role CT_QUICKSTART_USER --format JSON -q "
SELECT TO_CHAR(logged_at, 'HH24:MI:SS') AS t, LEFT(msg, 2500) AS m
FROM CT_QUICKSTART_DB.AGT.RUN_LOG WHERE run_tag = '$2' ORDER BY logged_at" 2>&1 \
  | python3 -c "import json,sys
try:
    rows=json.load(sys.stdin)
except Exception as e:
    print('log read failed', e); sys.exit()
for r in rows: print(r['T'], r['M'])"
