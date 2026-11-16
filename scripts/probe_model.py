# /// script
# requires-python = ">=3.11"
# dependencies = ["cortex-training @ https://github.com/snowflakedb/cortex-training/archive/b4c26d57bcd741dce4cd63a6ed8f51cff6ba8829.zip"]
# ///
"""Check whether Cortex Training accepts a model on this account's GPUs.

  probe_model.py MODEL [N_GPUS] [WAIT_SECONDS]

Submits a training-only LoRA job, waits until it runs or fails, prints the outcome,
and always cancels the job.
"""
import os
import sys
import time

from cortex_training.client import CortexTrainingClient

HOST = os.environ["SNOWFLAKE_HOST"]  # <org>-<account>.snowflakecomputing.com
model = sys.argv[1]
n_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 8
wait_s = int(sys.argv[3]) if len(sys.argv) > 3 else 1800

pat = open(os.path.expanduser("~/.snowflake/agt_lab/pat")).read().strip()
client = CortexTrainingClient.from_pat(HOST, pat, "CT_QUICKSTART_DB", "LAB")
lora = {"peft_type": "Lora", "r": 8, "lora_alpha": 8, "lora_dropout": 0.0, "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}
body = {"hardware": "H200", "idle_timeout_seconds": 600, "sub_job_configs": [
    {"job_type": "training", "model_name": model, "dtype": "bfloat16", "seed": 42,
     "training_config": {"n_gpus": n_gpus, "max_seq_len": 4096, "train_batch_size": 1,
                         "model_provider": "huggingface",
                         "optimizer": {"name": "AdamW", "lr": 1e-4},
                         "ds_config": {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}},
                         "peft_config": lora}}]}

print("capacity", client.get_capacity(hardware="H200"))
try:
    job_id = client.create_job_from_body(body)["job_id"]
except Exception as e:
    resp = getattr(e, "response", None)
    print("REJECTED at submit:", type(e).__name__, str(e)[:500], (resp.text[:1500] if resp is not None else ""))
    sys.exit(0)
print("job", job_id)
try:
    t0, last = time.time(), None
    while time.time() - t0 < wait_s:
        job = client.get_job(job_id)
        st = str(job.get("status", "")).lower().removeprefix("job_state_")
        if st != last:
            print(f"{time.time() - t0:.0f}s status={st} {str({k: v for k, v in job.items() if k not in ('body',)})[:800]}")
            last = st
        if st in ("running", "failed", "cancelled", "terminated", "done"):
            break
        time.sleep(20)
finally:
    try:
        client.cancel_job(job_id)
        print("cancelled", job_id)
    except Exception as e:
        print("cancel:", e)
