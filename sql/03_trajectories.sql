-- Build training data directly from AGENT_TRACE_TABLE. Nothing is exported.
--
-- TRAJECTORIES: one row per agent rollout (TruLens record), snapshotted so training
--   is reproducible and independent of event-table retention.
--   * identity (task_id, split, conversation_id) from the TruLens record_root span,
--     recorded from the LangGraph config metadata;
--   * eval scores from TruLens eval_root spans (eval.target_record_id = record_id);
--   * the full conversation from the LAST gateway chat span of that conversation:
--     its gen_ai.input.messages already contains every earlier turn and tool result,
--     and gen_ai.output.messages is the final assistant turn.
-- TRAINING_SET: the best train-split rollouts, at most 2 per task.

USE ROLE CT_QUICKSTART_USER;
USE SCHEMA CT_QUICKSTART_DB.AGT;
USE WAREHOUSE CT_QUICKSTART_WH;

SET RUN_TAG = 'hard-opus-train-v2';

CREATE TABLE IF NOT EXISTS TRAJECTORIES (
  run_tag STRING, record_id STRING, task_id STRING, split STRING, rollout INT,
  conversation_id STRING, model STRING, n_chat_calls INT, input_tokens INT, output_tokens INT,
  system_instructions VARIANT, input_messages VARIANT, output_messages VARIANT,
  finish_reason STRING, any_http_error BOOLEAN, scores OBJECT, eval_span_counts OBJECT,
  snapshotted_at TIMESTAMP_LTZ
);

DELETE FROM TRAJECTORIES WHERE run_tag = $RUN_TAG;

INSERT INTO TRAJECTORIES
WITH ev AS (
  SELECT * FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
  WHERE start_timestamp >= DATEADD('day', -7, CURRENT_TIMESTAMP())
),
roots AS (
  SELECT record_attributes:"ai.observability.record_id"::string AS record_id,
         PARSE_JSON(record_attributes:"ai.observability.call.kwargs.config"::string):metadata AS meta
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'record_root'
    AND record_attributes:"ai.observability.app_version"::string = $RUN_TAG
    AND record:"name"::string = 'langgraph.graph.state.CompiledStateGraph.invoke'
),
evals AS (
  SELECT record_attributes:"ai.observability.eval.target_record_id"::string AS record_id,
         record_attributes:"ai.observability.eval_root.metric_name"::string AS metric,
         AVG(record_attributes:"ai.observability.eval_root.score"::float) AS score,
         COUNT(*) AS n
  FROM ev
  WHERE record_attributes:"ai.observability.span_type"::string = 'eval_root'
    AND record_attributes:"ai.observability.app_version"::string = $RUN_TAG
  GROUP BY 1, 2
),
eval_obj AS (
  SELECT record_id, OBJECT_AGG(metric, score::variant) AS scores, OBJECT_AGG(metric, n::variant) AS counts
  FROM evals GROUP BY 1
),
chat AS (
  SELECT record_attributes:"gen_ai.conversation.id"::string AS conversation_id,
         start_timestamp,
         record_attributes:"gen_ai.request.model"::string AS model,
         TRY_TO_NUMBER(record_attributes:"gen_ai.usage.input_tokens"::string) AS input_tokens,
         TRY_TO_NUMBER(record_attributes:"gen_ai.usage.output_tokens"::string) AS output_tokens,
         TRY_TO_NUMBER(record_attributes:"http.status_code"::string) AS http_status,
         TRY_PARSE_JSON(record_attributes:"gen_ai.system_instructions"::string) AS sys,
         TRY_PARSE_JSON(record_attributes:"gen_ai.input.messages"::string) AS inp,
         TRY_PARSE_JSON(record_attributes:"gen_ai.output.messages"::string) AS outp
  FROM ev
  WHERE scope:"name"::string = 'aigateway/tracing'
    AND record:"kind"::string = 'SPAN_KIND_SERVER'
    AND record_attributes:"gen_ai.conversation.id"::string LIKE $RUN_TAG || ':%'
),
conv AS (
  SELECT conversation_id,
         MAX_BY(model, start_timestamp) AS model,
         COUNT(*) AS n_chat_calls,
         SUM(input_tokens) AS input_tokens,
         SUM(output_tokens) AS output_tokens,
         MAX_BY(sys, start_timestamp) AS sys,
         MAX_BY(inp, start_timestamp) AS inp,
         MAX_BY(outp, start_timestamp) AS outp,
         BOOLOR_AGG(http_status >= 400) AS any_http_error
  FROM chat GROUP BY 1
)
SELECT $RUN_TAG, r.record_id, r.meta:task_id::string, r.meta:split::string, r.meta:rollout::int,
       r.meta:conversation_id::string, c.model, c.n_chat_calls, c.input_tokens, c.output_tokens,
       c.sys, c.inp, c.outp, c.outp[0]:finish_reason::string, c.any_http_error,
       e.scores, e.counts, CURRENT_TIMESTAMP()
FROM roots r
LEFT JOIN conv c ON c.conversation_id = r.meta:conversation_id::string
LEFT JOIN eval_obj e ON e.record_id = r.record_id;

-- Stage counts for the selection funnel.
CREATE OR REPLACE VIEW TRAINING_FUNNEL AS
SELECT run_tag,
       COUNT(*)                                                                AS rollouts,
       COUNT_IF(split = 'train')                                               AS train_split,
       COUNT_IF(split = 'train' AND input_messages IS NOT NULL)                AS with_payload,
       COUNT_IF(split = 'train' AND input_messages IS NOT NULL AND NOT any_http_error
                AND finish_reason = 'stop')                                    AS clean_finish,
       COUNT_IF(split = 'train' AND input_messages IS NOT NULL AND NOT any_http_error
                AND finish_reason = 'stop' AND scores:correctness::float = 1)  AS correct,
       COUNT_IF(split = 'train' AND input_messages IS NOT NULL AND NOT any_http_error
                AND finish_reason = 'stop' AND scores:correctness::float = 1
                AND scores:answer_relevance::float >= 0.67)                    AS correct_and_relevant
FROM TRAJECTORIES GROUP BY 1;

-- Best rollouts: correct, relevant, cleanly finished; prefer fewer model calls and tokens.
CREATE OR REPLACE VIEW TRAINING_SET AS
SELECT * FROM (
  SELECT t.*,
         ROW_NUMBER() OVER (PARTITION BY run_tag, task_id
                            ORDER BY scores:answer_relevance::float DESC, n_chat_calls, output_tokens) AS rank_in_task
  FROM TRAJECTORIES t
  WHERE split = 'train'
    AND input_messages IS NOT NULL AND output_messages IS NOT NULL
    AND NOT any_http_error AND finish_reason = 'stop'
    AND scores:correctness::float = 1
    AND scores:answer_relevance::float >= 0.67
)
WHERE rank_in_task <= 2;

SELECT * FROM TRAINING_FUNNEL WHERE run_tag = $RUN_TAG;
