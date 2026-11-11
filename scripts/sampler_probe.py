# /// script
# requires-python = ">=3.11"
# dependencies = ["cortex-training @ https://github.com/snowflakedb/cortex-training/archive/b4c26d57bcd741dce4cd63a6ed8f51cff6ba8829.zip",
#                 "transformers==4.57.6", "jinja2>=3.1"]
# ///
"""Check that a sampling job loaded from a checkpoint generates from pre-rendered token ids.

  sampler_probe.py JOB_ID CHECKPOINT_ID

Workaround for the deployment chat endpoint failing server-side. Always cancels the job.
"""
import json
import os
import sys
import time

from cortex_training.client import CortexTrainingClient
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import agent_lib as L  # noqa: E402

HOST = "sfdevrel-enterprise.snowflakecomputing.com"
src_job, src_ckpt = sys.argv[1], sys.argv[2]
pat = open(os.path.expanduser("~/.snowflake/agt_lab/pat")).read().strip()
client = CortexTrainingClient.from_pat(HOST, pat, "CT_QUICKSTART_DB", "LAB", poll_timeout=1800)

lora = {"peft_type": "Lora", "r": 32, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
body = {"hardware": "H200", "idle_timeout_seconds": 600, "sub_job_configs": [
    {"job_type": "sampling", "model_name": "Qwen/Qwen3-8B", "dtype": "bfloat16", "seed": 42,
     "source_checkpoint_info": {"checkpoint_id": src_ckpt, "source_job_id": src_job},
     "inference_config": {"max_seq_len": 8192, "n_gpus": 4,
                          "vllm_config": {"tensor_parallel_size": 4, "gpu_memory_utilization": 0.8},
                          "peft_config": lora}}]}

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision="b968826d9c46dd6066d109eabc6255188de91218")
TOOLS = [
    {"type": "function", "function": {"name": "sql_query", "description": "Run a read-only SQLite query and return up to 50 rows as JSON.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression with + - * / ( ) and decimals.",
                                      "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
msgs = [{"role": "system", "content": L.SYSTEM_PROMPT_HARD + " /no_think"},
        {"role": "user", "content": "What was the net revenue from Toys products sold to East region customers in 2026-02? Round to 2 decimals."}]
ids = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=True, add_generation_prompt=True, enable_thinking=False)

job_id = client.create_job_from_body(body)["job_id"]
print("job", job_id, flush=True)
try:
    t0 = time.time()
    client.wait_for_job(job_id)
    print(f"running after {time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    rid = client.generate(job_id, [list(ids)], sampling_params={"max_tokens": 512, "temperature": 0.7})
    res = client.poll_request(job_id, rid)
    print(f"generate {time.time() - t1:.1f}s keys={list(res.keys())}")
    print(json.dumps(res, default=str)[:1500])
finally:
    client.cancel_job(job_id)
    print("cancelled", job_id)
