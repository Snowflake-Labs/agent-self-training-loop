-- Final comparison, read straight from AGENT_TRACE_TABLE: TruLens EVAL spans for each
-- held-out run (same harness, same tasks, same evaluators), plus gateway/agent cost signals.
USE ROLE CT_QUICKSTART_USER;
USE WAREHOUSE CT_QUICKSTART_WH;

WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
  WHERE start_timestamp >= DATEADD('day', -3, CURRENT_TIMESTAMP())
    AND record_attributes:"ai.observability.app_name"::string = 'retail_agent'
    AND record_attributes:"ai.observability.app_version"::string IN
        ('hard-final-opus', 'hard-final-qwen-base', 'hard-final-qwen-tuned')
),
records AS (
  SELECT record_attributes:"ai.observability.app_version"::string AS run,
         record_attributes:"ai.observability.record_id"::string   AS record_id,
         DATEDIFF('millisecond', start_timestamp, timestamp)       AS latency_ms
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'record_root'
    AND record:"name"::string = 'langgraph.graph.state.CompiledStateGraph.invoke'
),
tool_calls AS (
  SELECT record_attributes:"ai.observability.record_id"::string AS record_id, COUNT(*) AS n_tool_calls
  FROM ev WHERE record_attributes:"ai.observability.span_type"::string = 'tool' GROUP BY 1
),
scores AS (
  SELECT record_attributes:"ai.observability.eval.target_record_id"::string AS record_id,
         record_attributes:"ai.observability.eval_root.metric_name"::string AS metric,
         AVG(record_attributes:"ai.observability.eval_root.score"::float)   AS score
  FROM ev WHERE record_attributes:"ai.observability.span_type"::string = 'eval_root' GROUP BY 1, 2
)
SELECT CASE r.run WHEN 'hard-final-opus' THEN '1 claude-opus-5 (teacher)'
                  WHEN 'hard-final-qwen-base' THEN '2 Qwen3-8B base'
                  ELSE '3 Qwen3-8B post-trained' END                         AS model,
       COUNT(DISTINCT r.record_id)                                           AS rollouts,
       ROUND(100 * AVG(IFF(s.metric = 'correctness', s.score, NULL)), 1)     AS correctness_pct,
       ROUND(AVG(IFF(s.metric = 'answer_relevance', s.score, NULL)), 3)      AS answer_relevance,
       ROUND(MEDIAN(r.latency_ms) / 1000, 1)                                 AS median_latency_s,
       ROUND(AVG(COALESCE(t.n_tool_calls, 0)), 2)                            AS avg_tool_calls
FROM records r
LEFT JOIN scores s ON s.record_id = r.record_id
LEFT JOIN tool_calls t ON t.record_id = r.record_id
GROUP BY 1 ORDER BY 1;
