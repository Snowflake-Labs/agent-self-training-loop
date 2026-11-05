-- Inspect what the smoke test landed in AGENT_TRACE_TABLE. Replace the run tag.
SET RUN_TAG = 'smoke-20261104T110504';
WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
  WHERE start_timestamp >= DATEADD('hour', -2, CURRENT_TIMESTAMP())
),
app_traces AS (
  SELECT DISTINCT trace:"trace_id"::string AS tid FROM ev
  WHERE resource_attributes:"ai.observability.app_version"::string = $RUN_TAG
)
SELECT scope:"name"::string                                 AS scope_name,
       record_attributes:"ai.observability.span_type"::string AS trulens_span_type,
       record:"kind"::string                                 AS kind,
       COUNT(*)                                              AS spans,
       COUNT(DISTINCT trace:"trace_id"::string)              AS traces,
       COUNT_IF(trace:"trace_id"::string IN (SELECT tid FROM app_traces)) AS spans_in_app_traces,
       COUNT_IF(record_attributes:"gen_ai.input.messages" IS NOT NULL) AS with_input_payload
FROM ev
WHERE resource_attributes:"ai.observability.app_version"::string = $RUN_TAG
   OR trace:"trace_id"::string IN (SELECT tid FROM app_traces)
   OR record_attributes:"gen_ai.conversation.id"::string LIKE $RUN_TAG || '%'
GROUP BY ALL ORDER BY 1, 2;
