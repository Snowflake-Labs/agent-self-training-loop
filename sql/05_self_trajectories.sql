-- Trajectories from the deployed model's own production traces.
--
-- A Cortex Training deployment is called directly, not through the gateway, so there are no
-- gateway chat payloads. The harness instead puts the rollout's messages on a TruLens span
-- (attributes agt.trajectory, agt.conversation_id; joined on conversation id); that span, the record_root and the eval spans all land in
-- AGENT_TRACE_TABLE. Same selection rules as 03_trajectories.sql.

USE ROLE CT_QUICKSTART_USER;
USE SCHEMA CT_QUICKSTART_DB.AGT;
USE WAREHOUSE CT_QUICKSTART_WH;

SET RUN_TAG = 'r1-prod';

ALTER TABLE TRAJECTORIES ADD COLUMN IF NOT EXISTS chat_messages VARIANT;

DELETE FROM TRAJECTORIES WHERE run_tag = $RUN_TAG;

INSERT INTO TRAJECTORIES (run_tag, record_id, task_id, split, rollout, conversation_id, model, n_chat_calls,
                          finish_reason, any_http_error, scores, eval_span_counts, snapshotted_at, chat_messages)
WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
  WHERE start_timestamp >= DATEADD('day', -7, CURRENT_TIMESTAMP())
    AND record_attributes:"ai.observability.app_version"::string = $RUN_TAG
),
roots AS (
  SELECT record_attributes:"ai.observability.record_id"::string AS record_id,
         PARSE_JSON(record_attributes:"ai.observability.call.kwargs.config"::string):metadata AS meta
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'record_root'
    AND record:"name"::string = 'langgraph.graph.state.CompiledStateGraph.invoke'
),
traj0 AS (
  SELECT record_attributes:"agt.conversation_id"::string AS conversation_id,
         TRY_PARSE_JSON(record_attributes:"agt.trajectory"::string) AS msgs
  FROM ev
  WHERE record_attributes:"agt.trajectory" IS NOT NULL
),
traj AS (
  SELECT t.conversation_id, ANY_VALUE(t.msgs) AS msgs, COUNT_IF(f.value:role::string = 'assistant') AS n_assistant
  FROM traj0 t, LATERAL FLATTEN(t.msgs) f GROUP BY 1
),
evals AS (
  SELECT record_attributes:"ai.observability.eval.target_record_id"::string AS record_id,
         record_attributes:"ai.observability.eval_root.metric_name"::string AS metric,
         AVG(record_attributes:"ai.observability.eval_root.score"::float) AS score,
         COUNT(*) AS n
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'eval_root'
  GROUP BY 1, 2
),
eval_obj AS (
  SELECT record_id, OBJECT_AGG(metric, score::variant) AS scores, OBJECT_AGG(metric, n::variant) AS counts
  FROM evals GROUP BY 1
)
SELECT $RUN_TAG, r.record_id, r.meta:task_id::string, r.meta:split::string, r.meta:rollout::int,
       r.meta:conversation_id::string, 'self',
       t.n_assistant,
       IFF(t.msgs[ARRAY_SIZE(t.msgs) - 1]:role::string = 'assistant'
           AND t.msgs[ARRAY_SIZE(t.msgs) - 1]:tool_calls IS NULL, 'stop', 'incomplete'),
       FALSE, e.scores, e.counts, CURRENT_TIMESTAMP(), t.msgs
FROM roots r
JOIN traj t ON t.conversation_id = r.meta:conversation_id::string
LEFT JOIN eval_obj e ON e.record_id = r.record_id;

-- Same rules as before; a row qualifies with either gateway payloads or a recorded trajectory.
CREATE OR REPLACE VIEW TRAINING_SET AS
SELECT * FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY run_tag, task_id
                            ORDER BY scores:answer_relevance::float DESC, n_chat_calls, output_tokens) AS rank_in_task
  FROM TRAJECTORIES t
  WHERE split = 'train'
    AND ((input_messages IS NOT NULL AND output_messages IS NOT NULL) OR chat_messages IS NOT NULL)
    AND NOT any_http_error AND finish_reason = 'stop'
    AND scores:correctness::float = 1
    AND scores:answer_relevance::float >= 0.67
)
WHERE rank_in_task <= 2;

SELECT run_tag, COUNT(*) rollouts, COUNT_IF(split='train') train,
       COUNT_IF(finish_reason='stop') finished, COUNT_IF(scores:correctness::float = 1) correct
FROM TRAJECTORIES WHERE run_tag = $RUN_TAG GROUP BY 1;
SELECT run_tag, COUNT(*) selected, COUNT(DISTINCT task_id) tasks FROM TRAINING_SET WHERE run_tag = $RUN_TAG GROUP BY 1;

-- Judge-only selection: no ground truth. Judges score 0-3, normalized to 0, 1/3, 2/3, 1;
-- the cutoffs below are 2/3 (>= 0.66) and 1/3 (>= 0.33), written below the stored 0.6667 / 0.3333. correctness is still recorded on every run but used
-- only to report how often the judges kept a wrong answer.
CREATE OR REPLACE VIEW TRAINING_SET_JUDGED AS
SELECT * FROM (
  SELECT t.*,
         (scores:answer_relevance::float + scores:logical_consistency::float + scores:execution_efficiency::float
          + scores:tool_selection::float + scores:tool_calling::float) / 5 AS judge_mean,
         ROW_NUMBER() OVER (PARTITION BY run_tag, task_id ORDER BY judge_mean DESC, n_chat_calls) AS rank_in_task
  FROM TRAJECTORIES t
  WHERE split = 'train'
    AND chat_messages IS NOT NULL
    AND NOT any_http_error AND finish_reason = 'stop'
    AND scores:answer_relevance::float     >= 0.66
    -- Strict on logical consistency: on round-1 production runs this raised precision on new
    -- question kinds from 68% to 89% (kept runs that were actually correct).
    AND scores:logical_consistency::float  = 1
    AND scores:tool_selection::float       >= 0.66
    AND scores:tool_calling::float         >= 0.66
    AND scores:execution_efficiency::float >= 0.33
)
WHERE rank_in_task <= 2;

-- Judge precision: of the runs the judges kept, how many were actually correct.
SELECT run_tag, COUNT(*) AS kept, COUNT(DISTINCT task_id) AS tasks,
       ROUND(100 * AVG(scores:correctness::float), 1) AS pct_kept_actually_correct
FROM TRAINING_SET_JUDGED WHERE run_tag = $RUN_TAG GROUP BY 1;
