# /// script
# requires-python = ">=3.11"
# dependencies = ["cortex-training @ https://github.com/snowflakedb/cortex-training/archive/b4c26d57bcd741dce4cd63a6ed8f51cff6ba8829.zip"]
# ///
"""Create, wait for, or drop a Cortex Training EVALUATION deployment.

  deploy.py base                         # base Qwen/Qwen3-8B
  deploy.py checkpoint JOB_ID CKPT_ID    # tuned checkpoint
  deploy.py drop DEPLOYMENT_ID
The deployment id is appended to ~/.snowflake/agt_lab/deployments.log before waiting,
so an interrupted wait never loses track of a GPU-holding resource.
"""
import os
import sys
import time

from cortex_training.client import CortexTrainingClient

HOST = "sfdevrel-enterprise.snowflakecomputing.com"
MODEL = "Qwen/Qwen3-8B"
LOG = os.path.expanduser("~/.snowflake/agt_lab/deployments.log")

pat = open(os.path.expanduser("~/.snowflake/agt_lab/pat")).read().strip()
client = CortexTrainingClient.from_pat(HOST, pat, "CT_QUICKSTART_DB", "LAB", poll_timeout=2400)
cmd = sys.argv[1]

if cmd == "drop":
    client.drop_model_deployment(sys.argv[2])
    print({"dropped": sys.argv[2]})
    sys.exit()

if cmd == "base":
    source = {"type": "base_model", "model": MODEL}
elif cmd == "checkpoint":
    source = {"type": "checkpoint", "job_id": sys.argv[2], "checkpoint_id": sys.argv[3]}
else:
    raise SystemExit(__doc__)

spec = {
    "deployment": {"class": "EVALUATION", "min_replicas": 1, "gpus_per_replica": 4, "gpu_type": "H200"},
    "source": source,
    "inference": {"max_seq_len": 8192},
}
print("capacity", client.get_capacity(hardware="H200"))
dep = client.create_model_deployment(spec)["deployment"]
with open(LOG, "a") as fh:
    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {cmd} {dep}\n")
print({"deployment": dep})
ready = client.wait_for_model_deployment(dep)
print({"deployment": dep, "status": ready.get("status"), "base_url": ready.get("base_url")})
