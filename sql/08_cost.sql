-- Inputs to the cost comparison. Needs a role with IMPORTED PRIVILEGES on the SNOWFLAKE database.
-- Rates: Snowflake Service Consumption Table (effective November 9, 2026), $2.00 per AI credit.
--   claude-sonnet-4-6 via Cortex Inference (REST API): 1.50 / 7.50 AI credits per 1M input / output tokens
--   Cortex Training: 4.80 AI credits per H200 GPU-hour

USE ROLE ACCOUNTADMIN;

-- 1. Billed gateway requests per model (glm-5.3 is the seed model)
SELECT f.key AS model, COUNT(*) AS requests,
       SUM(OPERATION_DETAILS[f.key]:input_tokens::number)  AS input_tokens,
       SUM(OPERATION_DETAILS[f.key]:output_tokens::number) AS output_tokens,
       ROUND(SUM(f.value::float), 2) AS ai_credits
FROM SNOWFLAKE.ACCOUNT_USAGE.AI_GATEWAY_USAGE_HISTORY, LATERAL FLATTEN(CREDITS_GRANULAR) f
WHERE start_time >= '2026-11-04'
GROUP BY 1 ORDER BY ai_credits DESC;

-- 2. Judge tokens (TruLens Cortex provider calls go through the Cortex REST API)
SELECT model_name, COUNT(*) AS calls,
       SUM(TOKENS_GRANULAR:input::number)  AS input_tokens,
       SUM(TOKENS_GRANULAR:output::number) AS output_tokens,
       ROUND((SUM(TOKENS_GRANULAR:input::number) * 1.50 + SUM(TOKENS_GRANULAR:output::number) * 7.50) / 1e6, 1) AS ai_credits
FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_REST_API_USAGE_HISTORY
WHERE model_name = 'claude-sonnet-4-6' AND start_time >= '2026-11-04'
GROUP BY 1;

-- 3. Number of judge evaluations, to turn (2) into a per-judge, per-trace average
USE ROLE CT_QUICKSTART_USER;
USE SCHEMA CT_QUICKSTART_DB.AGT;
USE WAREHOUSE CT_QUICKSTART_WH;
SELECT COUNT(*) AS judge_evaluations,
       COUNT(DISTINCT record_attributes:"ai.observability.eval.target_record_id"::string) AS judged_runs
FROM TABLE(AGENT_TRACE_TABLE('SNOWFLAKE'))
WHERE start_timestamp >= '2026-11-04'
  AND record_attributes:"ai.observability.span_type"::string = 'eval_root';

-- 4. Training job sizes (examples = one per assistant turn of a kept trace)
SELECT run_tag, source_run_tag, examples, steps, created_at
FROM TRAINING_RUNS
WHERE run_tag IN ('jt-glm', 'jt-own1', 'jt-own2')
ORDER BY created_at;
