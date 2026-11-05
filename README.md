# Post-train on AI Gateway traces

Agent trace distillation inside Snowflake. A [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) agent runs through [Cortex AI Gateway](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-ai-gateway) with a frontier model, [TruLens](https://www.trulens.org/) scores every run, and [Cortex Training](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-training) post-trains [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) on the best-scoring traces. Trajectories are selected with SQL over [`AGENT_TRACE_TABLE`](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-ai-gateway/observability), and trace data stays in the Snowflake account.

## Result

49 held-out tasks, 2 rollouts each, same agent harness and same TruLens evaluators for every model. Numbers come from TruLens eval spans in `AGENT_TRACE_TABLE` ([`sql/04_compare.sql`](sql/04_compare.sql)).

| Model | Correct | Answer relevance | Median latency | Avg tool calls |
|---|---|---|---|---|
| claude-opus-5 (teacher) | 100.0% | 0.997 | 10.8 s | 2.71 |
| Qwen3-8B base | 13.7% | 0.708 | 2.8 s | 4.97 |
| Qwen3-8B post-trained | 90.8% | 0.968 | 2.8 s | 2.41 |

Training data: 372 Opus trajectories (2 per train task, 186 tasks), split into 1,031 per-turn examples. LoRA r=32, 2 epochs, 258 steps, loss 1.74 to about 0.11. One synthetic task family, one training run, no hyperparameter search.

## How it works

```mermaid
flowchart LR
    subgraph acct [Snowflake account]
        direction LR

        subgraph collect [1. Collect and evaluate]
            direction TB
            tasks["Task suite with SQL ground truth"]
            agent["LangGraph agent + TruLens (Code Bundle, CPU pool)"]
            tasks --> agent
        end

        subgraph gw [Cortex AI Gateway]
            teacher["claude-opus-5 (teacher)"]
        end

        judge["Cortex AI: claude-sonnet-4-6 (TruLens judge)"]

        traces[("AGENT_TRACE_TABLE: gateway chat spans with payloads, TruLens app and EVAL spans")]

        subgraph select [2. Select in SQL]
            direction TB
            traj["AGT.TRAJECTORIES"]
            tset["AGT.TRAINING_SET: correct, relevant, best 2 per task"]
            traj --> tset
        end

        subgraph train [3. Post-train]
            direction TB
            fmt["Code Bundle: Qwen chat template, per-turn loss mask"]
            ct["Cortex Training: Qwen3-8B LoRA on 8 H200"]
            ckpt["Checkpoint"]
            fmt --> ct --> ckpt
        end

        dep["EVALUATION deployment (OpenAI-compatible)"]
    end

    agent -->|"chat calls, conversation-id header"| teacher
    agent -->|"feedback calls via Snowpark"| judge
    teacher -->|"chat spans"| traces
    agent -->|"OTLP: app and EVAL spans"| traces
    traces --> traj
    tset --> fmt
    ckpt --> dep
    dep -.->|"held-out eval with the same harness"| agent
```

- TruLens normally exports to its own store. `make_tee_exporter` in [`agent_lib.py`](agent_lib.py) sends each span to both the local TruLens store, which feedback computation reads, and the gateway's OTLP endpoint `<gateway>/telemetry/v1/traces` ([send client traces to the gateway](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-ai-gateway/observability)). App spans, tool spans and `eval_root`/`eval` spans land in `AGENT_TRACE_TABLE`, which uses the standard [event table columns](https://docs.snowflake.com/en/developer-guide/logging-tracing/event-table-columns).
- Gateway chat spans carry the full prompts and responses (`logging.capture_payload.request_response: true`, set in [`sql/01_gateway_spec.sql`](sql/01_gateway_spec.sql)). The last chat span of a conversation holds the whole trajectory as [OpenTelemetry GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) message parts.
- Gateway spans join TruLens records on the `x-snowflake-ai-gateway-conversation-id` header. The same id is passed in the LangGraph `config` metadata and recorded on the TruLens `record_root` span. Eval spans join on `ai.observability.eval.target_record_id`. The joins are in [`sql/03_trajectories.sql`](sql/03_trajectories.sql).
- Feedbacks: `correctness`, which checks the final answer against SQL ground truth in code, and `answer_relevance`, judged by `claude-sonnet-4-6` through the TruLens Cortex provider. Snowflake's own TruLens integration is described in [AI Observability](https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability).
- The task suite (`agent_lib.suite("hard")`) is a synthetic retail-finance SQLite database whose business rules exist only in the system prompt (shipped orders only, net of discounts and returns, region from the customer). Base Qwen3-8B scores 94% on the simpler `basic` suite, so `hard` is the one used for the comparison.

## Files

| Path | Purpose |
|---|---|
| [`agent_lib.py`](agent_lib.py) | Task suites, LangGraph agent (gateway, or a Cortex Training deployment), tee exporter, TruLens feedbacks |
| [`bundle/batch_run.py`](bundle/batch_run.py) | [Code Bundle](https://docs.snowflake.com/en/sql-reference/sql/create-code-bundle) entrypoint: rollouts with TruLens, feedback computation, run index in `AGT.ROLLOUT_INDEX` |
| [`bundle/train_from_traces.py`](bundle/train_from_traces.py) | Code Bundle entrypoint: `TRAINING_SET` to Qwen chat format to Cortex Training LoRA, then a weights-only checkpoint |
| [`sql/01_gateway_spec.sql`](sql/01_gateway_spec.sql) | Gateway spec with client telemetry and payload capture |
| [`sql/02_external_access.sql`](sql/02_external_access.sql) | [External access integration](https://docs.snowflake.com/en/developer-guide/external-network-access/external-network-access-overview) limited to the account host |
| [`sql/03_trajectories.sql`](sql/03_trajectories.sql) | `TRAJECTORIES`, `TRAINING_FUNNEL`, `TRAINING_SET` built from `AGENT_TRACE_TABLE` |
| [`sql/04_compare.sql`](sql/04_compare.sql) | Final comparison from TruLens eval spans |
| [`scripts/mint_pat.py`](scripts/mint_pat.py) | Role-restricted [PAT](https://docs.snowflake.com/en/user-guide/programmatic-access-tokens) into a Snowflake secret, using a temporary key pair |
| [`scripts/deploy.py`](scripts/deploy.py) | Create or drop a Cortex Training EVALUATION deployment (base model or checkpoint) |
| [`scripts/smoke_test.py`](scripts/smoke_test.py) | One local rollout to check spans and eval spans before the batch run |

## Run it

Prerequisites: a Snowflake account with an AI Gateway, Cortex Training enabled (including model deployments), a CPU [compute pool](https://docs.snowflake.com/en/developer-guide/snowpark-container-services/working-with-compute-pool), a warehouse, and a role. The scripts use `CT_QUICKSTART_USER`, `CT_QUICKSTART_DB.AGT`, `CT_QUICKSTART_CPU` (3 nodes) and `CT_QUICKSTART_WH`, with the CLI connection `DEVREL_ENTERPRISE`; change these for your account. The `CT_QUICKSTART_DOWNLOAD_EAI` integration (PyPI, GitHub, Hugging Face) and the `CT_ACCOUNT_RULE` network rule come from the Cortex Training quickstart setup.

```bash
C="-c DEVREL_ENTERPRISE --role CT_QUICKSTART_USER --database CT_QUICKSTART_DB --schema AGT --warehouse CT_QUICKSTART_WH"

# 1. Gateway logging, PAT secret, external access
snow sql -c DEVREL_ENTERPRISE -f sql/01_gateway_spec.sql
uv run scripts/mint_pat.py
snow sql -c DEVREL_ENTERPRISE -f sql/02_external_access.sql

# 2. Bundles (agent_lib.py is copied in at build time)
cp agent_lib.py bundle/
snow bundle create AGT_BATCH --source bundle --overwrite $C
snow bundle create AGT_TRAIN --source bundle --overwrite $C

# 3. Teacher traces on the train split
snow bundle execute AGT_BATCH --entrypoint batch_run.py --async $C -- \
  --suite hard --run-tag hard-opus-train-v2 --splits train --rollouts 3 --workers 10

# 4. Select trajectories (set RUN_TAG in the file)
snow sql -c DEVREL_ENTERPRISE -f sql/03_trajectories.sql

# 5. Train (all 8 GPUs must be free)
snow bundle execute AGT_TRAIN --entrypoint train_from_traces.py --async $C -- \
  --run-tag train-hard-v1 --source-run-tag hard-opus-train-v2 --suite hard --epochs 2 --batch-size 8 --lr 1e-4

# 6. Evaluate base, tuned and teacher on the held-out split
uv run scripts/deploy.py base                      # or: checkpoint <JOB_ID> <CHECKPOINT_ID>
snow bundle execute AGT_BATCH --entrypoint batch_run.py --async $C -- \
  --suite hard --run-tag hard-final-qwen-tuned --splits heldout --rollouts 2 --workers 8 \
  --model <DEPLOYMENT_ID> --base-url https://<account-host>/api/v2/databases/CT_QUICKSTART_DB/schemas/LAB/cortex-training/inference/v1
snow sql -c DEVREL_ENTERPRISE -f sql/04_compare.sql
uv run scripts/deploy.py drop <DEPLOYMENT_ID>
```

Progress for any run: [`scripts/watch.sh`](scripts/watch.sh) `<query_id> <run_tag>` (bundle output goes to `AGT.RUN_LOG`). Bundles run with [`EXECUTE CODE BUNDLE`](https://docs.snowflake.com/en/sql-reference/sql/execute-code-bundle) under the hood.

## Notes and limits

- [`traceparent`](https://www.w3.org/TR/trace-context/) propagation through langchain-openai did not join client and gateway spans under one `trace_id`, so they are correlated by conversation id.
- Cortex Training deployments are EVALUATION class. They hold their GPUs until dropped. The deployment API is in the [Cortex Training SDK](https://github.com/snowflakedb/cortex-training).
- Call the gateway on the `<org>-<account>.snowflakecomputing.com` host. The `<org>-<account_name>` host that `SHOW AI GATEWAYS` returns fails Python TLS hostname checks when the account name contains an underscore.
- Qwen3's chat template renders past assistant turns without the empty `<think></think>` block, so training uses one example per assistant turn, rendered exactly as the model sees it at that step.
- Payload capture stores full prompts and responses in the event table. The tasks here are synthetic.
- The teacher's provider terms may restrict training other models on its outputs. Check them before publishing a model trained this way; `agent_lib.TEACHER_MODEL` can be any gateway model, for example `glm-5.3`.

## Cleanup

[`sql/00_restore_gateway_spec.sql`](sql/00_restore_gateway_spec.sql) restores the original gateway spec. Drop deployments with `scripts/deploy.py drop`, remove the PAT (needs a non-PAT session), and suspend the pool and warehouse.

## Resources

- [Cortex AI Gateway](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-ai-gateway) and [its observability and `AGENT_TRACE_TABLE`](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-ai-gateway/observability)
- [Cortex Training](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-training) and the [Cortex Training SDK](https://github.com/snowflakedb/cortex-training)
- [AI Observability with Snowflake Cortex](https://docs.snowflake.com/en/user-guide/snowflake-cortex/ai-observability)
- [TruLens](https://www.trulens.org/) ([GitHub](https://github.com/truera/trulens))
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [CREATE CODE BUNDLE](https://docs.snowflake.com/en/sql-reference/sql/create-code-bundle) and [EXECUTE CODE BUNDLE](https://docs.snowflake.com/en/sql-reference/sql/execute-code-bundle)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
