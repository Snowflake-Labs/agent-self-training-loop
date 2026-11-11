-- Held-out comparison across rounds, by old vs new question kinds, from TruLens eval spans
-- in AGENT_TRACE_TABLE. Task identity comes from the record_root span's LangGraph metadata.

USE ROLE CT_QUICKSTART_USER;
USE SCHEMA CT_QUICKSTART_DB.AGT;
USE WAREHOUSE CT_QUICKSTART_WH;

WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
  WHERE start_timestamp >= DATEADD('day', -7, CURRENT_TIMESTAMP())
    AND record_attributes:"ai.observability.app_version"::string
        IN ('hard-final-opus', 'opus-eval-new', 'hard-final-qwen-base', 'r1-eval', 'r2-eval', 'r3-eval',
            'glm-eval-v2', 'gr1-eval', 'jglm-eval', 'jown1-eval', 'jown2-eval')
),
roots AS (
  SELECT record_attributes:"ai.observability.app_version"::string AS run_tag,
         record_attributes:"ai.observability.record_id"::string AS record_id,
         PARSE_JSON(record_attributes:"ai.observability.call.kwargs.config"::string):metadata:task_id::string AS task_id
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'record_root'
    AND record:"name"::string = 'langgraph.graph.state.CompiledStateGraph.invoke'
),
scores AS (
  SELECT record_attributes:"ai.observability.eval.target_record_id"::string AS record_id,
         AVG(IFF(record_attributes:"ai.observability.eval_root.metric_name"::string = 'correctness',
                 record_attributes:"ai.observability.eval_root.score"::float, NULL)) AS correct,
         AVG(IFF(record_attributes:"ai.observability.eval_root.metric_name"::string = 'answer_relevance',
                 record_attributes:"ai.observability.eval_root.score"::float, NULL)) AS relevance
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'eval_root'
  GROUP BY 1
)
SELECT r.run_tag,
       IFF(STARTSWITH(r.task_id, 'n'), 'new kinds', 'round-1 kinds') AS kinds,
       COUNT(*) AS rollouts,
       ROUND(100 * AVG(s.correct), 1) AS pct_correct,
       ROUND(AVG(s.relevance), 3) AS relevance
FROM roots r JOIN TASKS_HARD2 t ON t.task_id = r.task_id AND t.split = 'heldout'
LEFT JOIN scores s ON s.record_id = r.record_id
GROUP BY 1, 2
ORDER BY 2, 1;
