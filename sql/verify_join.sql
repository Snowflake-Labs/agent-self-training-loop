-- One row per TruLens record for a run: identity from the record_root config kwarg,
-- eval scores via eval.target_record_id, gateway chat spans via conversation id.
SET RUN_TAG = 'smoke-20261104T110929';
WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
  WHERE start_timestamp >= DATEADD('hour', -3, CURRENT_TIMESTAMP())
),
roots AS (
  SELECT record_attributes:"ai.observability.record_id"::string AS record_id,
         PARSE_JSON(record_attributes:"ai.observability.call.kwargs.config"::string):metadata AS meta
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'record_root'
    AND record_attributes:"ai.observability.app_version"::string = $RUN_TAG
),
evals AS (
  SELECT record_attributes:"ai.observability.eval.target_record_id"::string AS record_id,
         record_attributes:"ai.observability.eval_root.metric_name"::string AS metric,
         record_attributes:"ai.observability.eval_root.score"::float        AS score
  FROM ev WHERE record_attributes:"ai.observability.span_type"::string = 'eval_root'
),
evals1 AS (
  SELECT record_id, metric, AVG(score) AS score, COUNT(*) AS n FROM evals GROUP BY 1, 2
),
gw AS (
  SELECT record_attributes:"gen_ai.conversation.id"::string AS conv, COUNT(*) AS chat_spans,
         COUNT_IF(record_attributes:"gen_ai.output.messages" IS NOT NULL) AS with_output
  FROM ev WHERE scope:"name"::string = 'aigateway/tracing' GROUP BY 1
)
SELECT r.record_id, r.meta:task_id::string AS task_id, r.meta:split::string AS split,
       OBJECT_AGG(e.metric, e.score::variant) AS scores, OBJECT_AGG(e.metric, e.n::variant) AS eval_span_counts, ANY_VALUE(g.chat_spans) AS gateway_chat_spans,
       ANY_VALUE(g.with_output) AS gateway_spans_with_payload
FROM roots r
LEFT JOIN evals1 e ON e.record_id = r.record_id
LEFT JOIN gw g ON g.conv = r.meta:conversation_id::string
GROUP BY 1, 2, 3;
