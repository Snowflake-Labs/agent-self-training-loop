-- Judge results per model version, from TruLens eval spans in AGENT_TRACE_TABLE.
--
-- 1. Held-out: share of runs that meet every judge cutoff in TRAINING_SET_JUDGED (the
--    reference-free signal the promotion gate uses), next to accuracy against the
--    reference answers (measured only to check the judges). Baseline, drift and overall.
-- 2. Collected runs on the training tasks: average score (out of 3) and share meeting
--    each judge's cutoff, per version.

USE ROLE CT_QUICKSTART_USER;
USE SCHEMA CT_QUICKSTART_DB.AGT;
USE WAREHOUSE CT_QUICKSTART_WH;

-- 1. Held-out judge pass rate and accuracy
WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
  WHERE start_timestamp >= DATEADD('day', -30, CURRENT_TIMESTAMP())
    AND record_attributes:"ai.observability.app_version"::string
        IN ('glm-eval-v2', 'jglm-eval', 'jown1-eval', 'jown2-eval')
),
roots AS (
  SELECT record_attributes:"ai.observability.app_version"::string AS run_tag,
         record_attributes:"ai.observability.record_id"::string AS record_id,
         PARSE_JSON(record_attributes:"ai.observability.call.kwargs.config"::string):metadata:task_id::string AS task_id
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'record_root'
    AND record:"name"::string = 'langgraph.graph.state.CompiledStateGraph.invoke'
),
sc AS (
  SELECT record_attributes:"ai.observability.eval.target_record_id"::string AS record_id,
         record_attributes:"ai.observability.eval_root.metric_name"::string AS metric,
         AVG(record_attributes:"ai.observability.eval_root.score"::float) AS score
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'eval_root'
  GROUP BY 1, 2
),
p AS (
  SELECT record_id,
         MAX(IFF(metric = 'answer_relevance', score, NULL))     AS rel,
         MAX(IFF(metric = 'logical_consistency', score, NULL))  AS lc,
         MAX(IFF(metric = 'execution_efficiency', score, NULL)) AS ee,
         MAX(IFF(metric = 'tool_selection', score, NULL))       AS ts,
         MAX(IFF(metric = 'tool_calling', score, NULL))         AS tc,
         MAX(IFF(metric = 'correctness', score, NULL))          AS correct
  FROM sc GROUP BY 1
)
SELECT r.run_tag,
       COALESCE(IFF(STARTSWITH(r.task_id, 'n'), 'drift', 'baseline'), 'overall') AS tasks,
       COUNT(*) AS runs,
       ROUND(100 * AVG(IFF(rel >= 0.66 AND lc = 1 AND ts >= 0.66 AND tc >= 0.66 AND ee >= 0.33, 1, 0)), 1) AS judge_pass_pct,
       ROUND(100 * AVG(correct), 1) AS accuracy_pct,
       COUNT(correct) AS runs_with_reference_score
FROM roots r
JOIN TASKS_HARD2 t ON t.task_id = r.task_id AND t.split = 'heldout'
LEFT JOIN p ON p.record_id = r.record_id
GROUP BY GROUPING SETS ((r.run_tag, IFF(STARTSWITH(r.task_id, 'n'), 'drift', 'baseline')), (r.run_tag))
ORDER BY 1, 2;

-- 2. Score distribution on the collected training-task runs
WITH t AS (
  SELECT * FROM TRAJECTORIES
  WHERE split = 'train' AND run_tag IN ('glm-seed-v2', 'jglm-prod', 'jown1-prod')
),
u AS (
  SELECT run_tag, f.key AS judge, f.value::float AS v
  FROM t, LATERAL FLATTEN(scores) f
  WHERE f.key IN ('answer_relevance', 'logical_consistency', 'execution_efficiency', 'tool_selection', 'tool_calling')
)
SELECT run_tag, judge, COUNT(*) AS runs,
       ROUND(3 * AVG(v), 2) AS avg_out_of_3,
       ROUND(100 * AVG(IFF(v >= CASE judge WHEN 'logical_consistency' THEN 1
                                           WHEN 'execution_efficiency' THEN 0.33
                                           ELSE 0.66 END, 1, 0)), 1) AS meets_cutoff_pct
FROM u
GROUP BY 1, 2
ORDER BY 2, 1;
