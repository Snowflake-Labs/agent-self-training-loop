"""Shared harness: synthetic retail DB, tasks with ground truth, LangGraph agent,
TruLens instrumentation that tees spans to the local TruLens store and to the
Cortex AI Gateway OTLP endpoint, and the feedback functions.

Used by both the local smoke test and the in-Snowflake batch run.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import uuid
from typing import Sequence

# The underscore account URL fails Python TLS hostname checks; the org-account host serves the same gateway.
GATEWAY = "https://sfdevrel-enterprise.snowflakecomputing.com/api/v2/aigateways/snowflake"
TEACHER_MODEL = "glm-5.3"  # seed model via the gateway; Qwen3-8B is the model improved round over round
JUDGE_MODEL = "claude-sonnet-4-6"
SYSTEM_PROMPT = (
    "You are a retail analytics agent. Answer the user's question using the tools. "
    "Use sql_query to read the SQLite database (tables: products(id, name, category, price, stock), "
    "orders(id, product_id, qty, order_date, region); order_date is 'YYYY-MM-DD'; revenue = qty * price). "
    "Use calculator for arithmetic you do not do in SQL. Check your result before answering. "
    "End your reply with a line 'FINAL ANSWER: <value>' where value is a number (no units or "
    "thousands separators) or a product name."
)

CATEGORIES = ["Outdoor", "Kitchen", "Office", "Toys", "Garden"]
REGIONS = ["North", "South", "East", "West"]
MONTHS = [f"2026-{m:02d}" for m in range(1, 7)]


# --------------------------------------------------------------------------- data
def build_db(seed: int = 7) -> sqlite3.Connection:
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER)")
    conn.execute("CREATE TABLE orders(id INTEGER PRIMARY KEY, product_id INTEGER, qty INTEGER, order_date TEXT, region TEXT)")
    adjectives = ["Alpine", "Brisk", "Cedar", "Delta", "Ember", "Fjord", "Granite", "Harbor"]
    pid = 0
    for cat in CATEGORIES:
        for adj in rng.sample(adjectives, 6):
            pid += 1
            conn.execute("INSERT INTO products VALUES (?,?,?,?,?)",
                         (pid, f"{adj} {cat[:-1] if cat.endswith('s') else cat} {pid}", cat,
                          round(rng.uniform(5, 250), 2), rng.randint(0, 400)))
    for oid in range(1, 1201):
        conn.execute("INSERT INTO orders VALUES (?,?,?,?,?)",
                     (oid, rng.randint(1, pid), rng.randint(1, 12),
                      f"{rng.choice(MONTHS)}-{rng.randint(1, 28):02d}", rng.choice(REGIONS)))
    conn.commit()
    return conn


def _one(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def make_tasks(conn: sqlite3.Connection, n: int = 240, seed: int = 11, heldout_frac: float = 0.2) -> list[dict]:
    """Questions with ground truth computed in SQL. Split is assigned by task_id."""
    rng = random.Random(seed)
    tasks, seen = [], set()
    while len(tasks) < n:
        kind = rng.choice(["rev_region_cat", "units_product_month", "avg_price_cat",
                           "low_stock_cat", "rev_diff_regions", "top_product_region"])
        cat, reg, reg2, month = rng.choice(CATEGORIES), rng.choice(REGIONS), rng.choice(REGIONS), rng.choice(MONTHS)
        pname = _one(conn, "SELECT name FROM products WHERE category=? ORDER BY id LIMIT 1 OFFSET ?", (cat, rng.randint(0, 5)))
        if kind == "rev_region_cat":
            q = f"What was the total revenue from {cat} products in the {reg} region? Round to 2 decimals."
            a = round(_one(conn, "SELECT COALESCE(SUM(o.qty*p.price),0) FROM orders o JOIN products p ON p.id=o.product_id WHERE p.category=? AND o.region=?", (cat, reg)), 2)
        elif kind == "units_product_month":
            q = f"How many units of '{pname}' were ordered in {month}?"
            a = _one(conn, "SELECT COALESCE(SUM(o.qty),0) FROM orders o JOIN products p ON p.id=o.product_id WHERE p.name=? AND substr(o.order_date,1,7)=?", (pname, month))
        elif kind == "avg_price_cat":
            q = f"What is the average list price of {cat} products? Round to 2 decimals."
            a = round(_one(conn, "SELECT AVG(price) FROM products WHERE category=?", (cat,)), 2)
        elif kind == "low_stock_cat":
            q = f"Which {cat} product has the lowest stock? Give the product name."
            a = _one(conn, "SELECT name FROM products WHERE category=? ORDER BY stock, id LIMIT 1", (cat,))
        elif kind == "rev_diff_regions":
            if reg == reg2:
                continue
            q = f"How much more revenue did {reg} generate than {reg2} in {month}? Negative if less. Round to 2 decimals."
            r = lambda g: _one(conn, "SELECT COALESCE(SUM(o.qty*p.price),0) FROM orders o JOIN products p ON p.id=o.product_id WHERE o.region=? AND substr(o.order_date,1,7)=?", (g, month))
            a = round(r(reg) - r(reg2), 2)
        else:
            q = f"Which product sold the most units in the {reg} region in {month}? Give the product name."
            a = _one(conn, "SELECT p.name FROM orders o JOIN products p ON p.id=o.product_id WHERE o.region=? AND substr(o.order_date,1,7)=? GROUP BY p.name ORDER BY SUM(o.qty) DESC, p.name LIMIT 1", (reg, month))
        if q in seen or a is None:
            continue
        seen.add(q)
        tid = f"t{len(tasks):04d}"
        split = "heldout" if (int(uuid.uuid5(uuid.NAMESPACE_URL, tid).hex, 16) % 100) < heldout_frac * 100 else "train"
        tasks.append({"task_id": tid, "kind": kind, "question": q, "answer": a, "split": split})
    return tasks


# --------------------------------------------------------------------------- hard suite
# Base Qwen3-8B scores ~94% on the suite above, so the frontier-vs-student comparison uses
# this one: a normalized schema whose business rules exist only in the prompt, and
# questions that need 3-5 joins, status/return handling and multi-step arithmetic.
SYSTEM_PROMPT_HARD = (
    "You are a retail finance analytics agent. Answer the user's question using the tools. "
    "Use sql_query to read the SQLite database. Tables: customers(id, name, region, segment); "
    "products(id, name, category, list_price); orders(id, customer_id, order_date, status) with order_date "
    "'YYYY-MM-DD' and status in ('shipped','cancelled','pending'); order_items(order_id, product_id, qty, "
    "unit_price, discount_pct); returns(order_id, product_id, qty_returned, return_date). "
    "Business rules: only orders with status 'shipped' count. A line's net value is "
    "qty * unit_price * (1 - discount_pct/100). Net revenue is the sum of line net values minus the "
    "value of returned units, where returned value is qty_returned * unit_price * (1 - discount_pct/100) "
    "of the matching order line. A sale's region and segment come from the customer. Months are by order_date. "
    "Use calculator for arithmetic you do not do in SQL. Check your result before answering. "
    "End your reply with a line 'FINAL ANSWER: <value>' where value is a number (no units, percent signs or "
    "thousands separators) or a name."
)
SYSTEM_PROMPTS = {"basic": SYSTEM_PROMPT, "hard": SYSTEM_PROMPT_HARD, "hard2": SYSTEM_PROMPT_HARD}
SEGMENTS = ["Consumer", "Business"]

_NET_LINES = """
WITH lines AS (
  SELECT o.id AS order_id, o.order_date, substr(o.order_date,1,7) AS month, c.id AS customer_id, c.name AS customer,
         c.region, c.segment, p.id AS product_id, p.name AS product, p.category, p.list_price,
         oi.qty, oi.unit_price, oi.discount_pct,
         oi.qty * oi.unit_price * (1 - oi.discount_pct/100.0) AS gross_net,
         COALESCE((SELECT SUM(r.qty_returned) FROM returns r WHERE r.order_id=o.id AND r.product_id=p.id), 0) AS qty_ret
  FROM orders o JOIN customers c ON c.id=o.customer_id JOIN order_items oi ON oi.order_id=o.id
  JOIN products p ON p.id=oi.product_id WHERE o.status='shipped'
), net AS (
  SELECT *, gross_net - qty_ret * unit_price * (1 - discount_pct/100.0) AS net_value FROM lines
)
"""


def build_db_hard(seed: int = 7) -> sqlite3.Connection:
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript("""
      CREATE TABLE customers(id INTEGER PRIMARY KEY, name TEXT, region TEXT, segment TEXT);
      CREATE TABLE products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, list_price REAL);
      CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT);
      CREATE TABLE order_items(order_id INTEGER, product_id INTEGER, qty INTEGER, unit_price REAL, discount_pct REAL);
      CREATE TABLE returns(order_id INTEGER, product_id INTEGER, qty_returned INTEGER, return_date TEXT);
    """)
    first = ["Acme", "Birch", "Cobalt", "Dune", "Echo", "Flint", "Grove", "Halo", "Iris", "Juno", "Kite", "Lumen"]
    second = ["Labs", "Supply", "Works", "Co", "Group", "Partners"]
    for cid in range(1, 61):
        conn.execute("INSERT INTO customers VALUES (?,?,?,?)",
                     (cid, f"{rng.choice(first)} {rng.choice(second)} {cid}", rng.choice(REGIONS), rng.choice(SEGMENTS)))
    adjectives = ["Alpine", "Brisk", "Cedar", "Delta", "Ember", "Fjord", "Granite", "Harbor"]
    pid = 0
    for cat in CATEGORIES:
        for adj in rng.sample(adjectives, 6):
            pid += 1
            conn.execute("INSERT INTO products VALUES (?,?,?,?)",
                         (pid, f"{adj} {cat[:-1] if cat.endswith('s') else cat} {pid}", cat, round(rng.uniform(5, 250), 2)))
    prices = {r[0]: r[1] for r in conn.execute("SELECT id, list_price FROM products")}
    for oid in range(1, 901):
        date = f"{rng.choice(MONTHS)}-{rng.randint(1, 28):02d}"
        status = rng.choices(["shipped", "cancelled", "pending"], [0.8, 0.12, 0.08])[0]
        conn.execute("INSERT INTO orders VALUES (?,?,?,?)", (oid, rng.randint(1, 60), date, status))
        for prod in rng.sample(range(1, pid + 1), rng.randint(1, 4)):
            qty = rng.randint(1, 10)
            unit = round(prices[prod] * rng.uniform(0.85, 1.05), 2)
            disc = rng.choice([0, 0, 0, 5, 10, 15, 20])
            conn.execute("INSERT INTO order_items VALUES (?,?,?,?,?)", (oid, prod, qty, unit, disc))
            if status == "shipped" and rng.random() < 0.12:
                conn.execute("INSERT INTO returns VALUES (?,?,?,?)",
                             (oid, prod, rng.randint(1, qty), f"{date[:7]}-28"))
            elif status != "shipped" and rng.random() < 0.05:
                # Returns on non-shipped orders are data noise that the rules exclude.
                conn.execute("INSERT INTO returns VALUES (?,?,?,?)", (oid, prod, 1, f"{date[:7]}-28"))
    conn.commit()
    return conn


def make_tasks_hard(conn: sqlite3.Connection, n: int = 240, seed: int = 13, heldout_frac: float = 0.2,
                    max_attempts: int = 20000) -> list[dict]:
    rng = random.Random(seed)
    q1 = lambda sql, p=(): conn.execute(_NET_LINES + sql, p).fetchone()[0]
    tasks, seen = [], set()
    kinds = ["net_rev_cat_region_month", "return_rate_cat_month", "mom_growth_segment", "second_customer_region_month",
             "aov_segment_region_month", "discount_gap_product_region", "churned_customers"]
    attempts = 0
    while len(tasks) < n and attempts < max_attempts:
        attempts += 1
        kind = rng.choice(kinds)
        cat, reg, seg = rng.choice(CATEGORIES), rng.choice(REGIONS), rng.choice(SEGMENTS)
        mi = rng.randint(0, len(MONTHS) - 2)
        month, nxt = MONTHS[mi], MONTHS[mi + 1]
        if kind == "net_rev_cat_region_month":
            q = f"What was the net revenue from {cat} products sold to {reg} region customers in {month}? Round to 2 decimals."
            a = q1("SELECT COALESCE(SUM(net_value),0) FROM net WHERE category=? AND region=? AND month=?", (cat, reg, month))
        elif kind == "return_rate_cat_month":
            q = f"What percentage of {cat} units shipped in {month} were returned? Round to 2 decimals."
            a = q1("SELECT 100.0*SUM(qty_ret)/NULLIF(SUM(qty),0) FROM net WHERE category=? AND month=?", (cat, month))
        elif kind == "mom_growth_segment":
            q = (f"By what percent did net revenue from {seg} customers change from {month} to {nxt}? "
                 "Negative if it fell. Round to 2 decimals.")
            a0 = q1("SELECT SUM(net_value) FROM net WHERE segment=? AND month=?", (seg, month))
            a1 = q1("SELECT SUM(net_value) FROM net WHERE segment=? AND month=?", (seg, nxt))
            a = None if not a0 else 100.0 * (a1 - a0) / a0
        elif kind == "second_customer_region_month":
            q = (f"Which customer had the second-highest net revenue in the {reg} region in {month}? "
                 "Give the customer name.")
            a = q1("SELECT customer FROM net WHERE region=? AND month=? GROUP BY customer_id "
                   "ORDER BY SUM(net_value) DESC, customer_id LIMIT 1 OFFSET 1", (reg, month))
        elif kind == "aov_segment_region_month":
            q = (f"What was the average net revenue per shipped order for {seg} customers in the {reg} region "
                 f"in {month}? Round to 2 decimals.")
            a = q1("SELECT SUM(net_value)/NULLIF(COUNT(DISTINCT order_id),0) FROM net WHERE segment=? AND region=? AND month=?",
                   (seg, reg, month))
        elif kind == "discount_gap_product_region":
            q = (f"Among {cat} products sold to {reg} region customers, which one has the largest gap between its "
                 "list price and its average realized price per unit sold (after discount, shipped orders, weighted "
                 "by quantity)? Give the product name.")
            a = q1("SELECT product FROM net WHERE category=? AND region=? GROUP BY product_id "
                   "ORDER BY MAX(list_price) - SUM(qty*unit_price*(1-discount_pct/100.0))/SUM(qty) DESC, product_id LIMIT 1",
                   (cat, reg))
        else:
            q = (f"How many {reg} region customers had at least one shipped order in {month} "
                 f"but no shipped orders in {nxt}?")
            a = q1("SELECT COUNT(*) FROM (SELECT DISTINCT customer_id FROM net WHERE region=? AND month=? "
                   "EXCEPT SELECT DISTINCT customer_id FROM net WHERE month=?)", (reg, month, nxt))
        if a is None or q in seen:
            continue
        if isinstance(a, float):
            a = round(a, 2)
        seen.add(q)
        tid = f"h{len(tasks):04d}"
        split = "heldout" if (int(uuid.uuid5(uuid.NAMESPACE_URL, tid).hex, 16) % 100) < heldout_frac * 100 else "train"
        tasks.append({"task_id": tid, "kind": kind, "question": q, "answer": a, "split": split})
    return tasks


NEW_KINDS = ["top_category_region_month", "return_value_segment_month", "category_share_region_month",
             "repeat_buyers_cat_region"]


def make_tasks_new(conn: sqlite3.Connection, n: int = 160, seed: int = 29, heldout_frac: float = 0.2,
                   max_attempts: int = 20000) -> list[dict]:
    """Question kinds added after round 1 (ids n0000...): the drift a deployed agent sees."""
    rng = random.Random(seed)
    q1 = lambda sql, p=(): conn.execute(_NET_LINES + sql, p).fetchone()[0]
    tasks, seen, attempts = [], set(), 0
    while len(tasks) < n and attempts < max_attempts:
        attempts += 1
        kind = rng.choice(NEW_KINDS)
        cat, reg, seg, month = rng.choice(CATEGORIES), rng.choice(REGIONS), rng.choice(SEGMENTS), rng.choice(MONTHS)
        if kind == "top_category_region_month":
            q = f"Which product category had the highest net revenue from {reg} region customers in {month}? Give the category."
            a = q1("SELECT category FROM net WHERE region=? AND month=? GROUP BY category "
                   "ORDER BY SUM(net_value) DESC, category LIMIT 1", (reg, month))
        elif kind == "return_value_segment_month":
            q = (f"What was the total value of units returned by {seg} customers on shipped orders placed in {month}, "
                 "at the discounted price paid? Round to 2 decimals.")
            a = q1("SELECT SUM(qty_ret*unit_price*(1-discount_pct/100.0)) FROM net WHERE segment=? AND month=?", (seg, month))
        elif kind == "category_share_region_month":
            q = (f"What percentage of the {reg} region's net revenue in {month} came from {cat} products? "
                 "Round to 2 decimals.")
            tot = q1("SELECT SUM(net_value) FROM net WHERE region=? AND month=?", (reg, month))
            part = q1("SELECT COALESCE(SUM(net_value),0) FROM net WHERE region=? AND month=? AND category=?", (reg, month, cat))
            a = None if not tot else 100.0 * part / tot
        else:
            q = (f"How many {reg} region customers bought {cat} products on shipped orders in at least two "
                 "different months?")
            a = q1("SELECT COUNT(*) FROM (SELECT customer_id FROM net WHERE region=? AND category=? "
                   "GROUP BY customer_id HAVING COUNT(DISTINCT month) >= 2)", (reg, cat))
        if a is None or q in seen:
            continue
        if isinstance(a, float):
            a = round(a, 2)
        seen.add(q)
        tid = f"n{len(tasks):04d}"
        split = "heldout" if (int(uuid.uuid5(uuid.NAMESPACE_URL, tid).hex, 16) % 100) < heldout_frac * 100 else "train"
        tasks.append({"task_id": tid, "kind": kind, "question": q, "answer": a, "split": split})
    return tasks


def suite(name: str):
    """(db, tasks, system_prompt) for a task suite. hard2 = the hard suite unchanged plus NEW_KINDS."""
    if name == "hard2":
        db = build_db_hard()
        return db, make_tasks_hard(db) + make_tasks_new(db), SYSTEM_PROMPT_HARD
    if name == "hard":
        db = build_db_hard()
        return db, make_tasks_hard(db), SYSTEM_PROMPT_HARD
    db = build_db()
    return db, make_tasks(db), SYSTEM_PROMPT

# --------------------------------------------------------------------------- scoring
_FINAL = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)


def extract_final(text) -> str | None:
    m = _FINAL.findall(content_text(text))
    return m[-1].strip().strip("*`'\". ") if m else None


def is_correct(final_text: str, expected) -> bool:
    got = extract_final(final_text)
    if got is None:
        return False
    if isinstance(expected, (int, float)):
        try:
            return abs(float(got.replace(",", "").replace("$", "")) - float(expected)) <= 0.011
        except ValueError:
            return False
    return got.lower() == str(expected).lower()


# --------------------------------------------------------------------------- agent
def run_sql(conn: sqlite3.Connection, query: str) -> str:
    """The sql_query tool. Deterministic for a given database, so stored runs can be replayed."""
    if not re.match(r"^\s*(select|with)\b", query, re.IGNORECASE):
        return "ERROR: only SELECT queries are allowed"
    try:
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description]
        return json.dumps([dict(zip(cols, r)) for r in cur.fetchmany(50)])
    except Exception as e:  # surfaced to the model
        return f"ERROR: {e}"


def run_calculator(expression: str) -> str:
    """The calculator tool."""
    if not re.fullmatch(r"[0-9.+\-*/() eE]+", expression):
        return "ERROR: unsupported characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - charset restricted above
    except Exception as e:
        return f"ERROR: {e}"


def tool_text(content) -> str:
    """A tool message's content as the model sees it."""
    return content if isinstance(content, str) else json.dumps(content)


# Tool schemas exactly as the trainer renders them (bundle/train_from_traces.py TOOLS).
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "sql_query", "description": "Run a read-only SQLite query and return up to 50 rows as JSON.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression with + - * / ( ) and decimals.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class Sampler:
    """A Cortex Training sampling job loaded from a checkpoint, called with pre-rendered token ids.

    Used instead of a model deployment's chat endpoint: the job takes the prompt exactly as the
    trainer rendered it, so no server-side chat template is involved."""

    MODEL = "Qwen/Qwen3-8B"
    TOKENIZER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

    def __init__(self, client, job_id: str):
        from transformers import AutoTokenizer

        self.client, self.job_id = client, job_id
        self.tok = AutoTokenizer.from_pretrained(self.MODEL, revision=self.TOKENIZER_REVISION)

    @classmethod
    def start(cls, client, source_job_id: str, checkpoint_id: str, n_gpus: int = 4, max_seq_len: int = 8192):
        lora = {"peft_type": "Lora", "r": 32, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
        body = {"hardware": "H200", "idle_timeout_seconds": 900, "sub_job_configs": [
            {"job_type": "sampling", "model_name": cls.MODEL, "dtype": "bfloat16", "seed": 42,
             "source_checkpoint_info": {"checkpoint_id": checkpoint_id, "source_job_id": source_job_id},
             "inference_config": {"max_seq_len": max_seq_len, "n_gpus": n_gpus,
                                  "vllm_config": {"tensor_parallel_size": n_gpus, "gpu_memory_utilization": 0.8},
                                  "peft_config": lora}}]}
        job_id = client.create_job_from_body(body)["job_id"]
        return cls(client, job_id)

    def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        ids = self.tok.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
        rid = self.client.generate(self.job_id, [list(ids)],
                                   sampling_params={"max_tokens": max_tokens, "temperature": temperature})
        return self.client.poll_request(self.job_id, rid)["results"][0]["text"]


def make_sampler_chat_model(sampler: Sampler, temperature: float, max_tokens: int = 1024):
    """LangChain chat model over a Sampler: renders the Qwen chat template locally and parses
    <tool_call> blocks into tool calls, the same format the model was trained on."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    def to_chat(m) -> dict:
        if m.type == "system":
            return {"role": "system", "content": content_text(m.content)}
        if m.type == "human":
            return {"role": "user", "content": content_text(m.content)}
        if m.type == "tool":
            return {"role": "tool", "content": tool_text(m.content)}
        msg = {"role": "assistant", "content": content_text(m.content).strip()}
        if m.tool_calls:
            msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                                 for c in m.tool_calls]
        return msg

    class SamplerChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "cortex-training-sampler"

        def bind_tools(self, tools, **kwargs):
            return self  # the tool schemas are fixed and rendered into every prompt

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            text = sampler.complete([to_chat(m) for m in messages], max_tokens, temperature)
            calls = []
            for i, block in enumerate(_TOOL_CALL.findall(text)):
                try:
                    c = json.loads(block)
                    calls.append({"name": c["name"], "args": c.get("arguments") or {}, "id": f"call_{uuid.uuid4().hex[:12]}_{i}"})
                except (ValueError, KeyError):
                    continue  # malformed call: left in the text, so the run fails visibly
            content = _TOOL_CALL.sub("", text).strip() if calls else text.strip()
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content, tool_calls=calls))])

    return SamplerChatModel()


def build_agent(conn: sqlite3.Connection, api_key: str, conversation_id: str,
                model: str = TEACHER_MODEL, base_url: str | None = None, temperature: float = 0.7,
                system_prompt: str = SYSTEM_PROMPT, sampler: Sampler | None = None):
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    @tool
    def sql_query(query: str) -> str:
        """Run a read-only SQLite query and return up to 50 rows as JSON."""
        return run_sql(conn, query)

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression with + - * / ( ) and decimals."""
        return run_calculator(expression)

    if sampler is not None:
        # The system prompt is passed as a message, so the template renders it as in training.
        llm = make_sampler_chat_model(sampler, temperature)
        return create_react_agent(llm, [sql_query, calculator], prompt=system_prompt + " /no_think")
    if model.startswith("claude"):
        # Claude models are served on the gateway's Anthropic Messages API, not /v1/chat/completions.
        from langchain_anthropic import ChatAnthropic

        # Opus 5 rejects `temperature` (400 "deprecated for this model"), so it is not sent.
        llm = ChatAnthropic(
            model=model, base_url=base_url or GATEWAY, api_key=api_key,
            max_retries=2, timeout=180, max_tokens=2048,
            default_headers={"x-snowflake-ai-gateway-conversation-id": conversation_id},
        )
    else:
        headers = {"x-snowflake-ai-gateway-conversation-id": conversation_id}
        prompt = system_prompt
        if model.startswith("sf-ct-"):
            # Cortex Training EVALUATION deployment (base or tuned Qwen3-8B), called directly.
            headers["X-Snowflake-Authorization-Token-Type"] = "PROGRAMMATIC_ACCESS_TOKEN"
            prompt = system_prompt + " /no_think"
        llm = ChatOpenAI(
            model=model, base_url=base_url or f"{GATEWAY}/v1", api_key=api_key,
            temperature=temperature, max_retries=8, timeout=180,  # gateway models are rate limited per account
            # Gateway models such as glm-5.3 reason before answering; the 8B deployment does not.
            max_completion_tokens=1024 if model.startswith("sf-ct-") else 4096,
            default_headers=headers,
        )
        return create_react_agent(llm, [sql_query, calculator], prompt=prompt)
    return create_react_agent(llm, [sql_query, calculator], prompt=system_prompt)


# --------------------------------------------------------------------------- tracing
def make_tee_exporter(local_connector, api_key: str):
    """Send every span to TruLens' own store (needed to compute feedbacks) and to the gateway."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from trulens.experimental.otel_tracing.core.exporter.connector import TruLensOtelSpanExporter

    class TeeExporter(SpanExporter):
        def __init__(self):
            self.local = TruLensOtelSpanExporter(local_connector)
            self.remote = OTLPSpanExporter(endpoint=f"{GATEWAY}/telemetry/v1/traces",
                                           headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
            self.remote_failures = 0
            self.remote_spans = 0

        def export(self, spans: Sequence) -> "SpanExportResult":
            local_result = self.local.export(spans)
            r = self.remote.export(spans)
            if r != SpanExportResult.SUCCESS:
                self.remote_failures += 1
            else:
                self.remote_spans += len(spans)
            return local_result

        def shutdown(self):
            self.local.shutdown()
            self.remote.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    return TeeExporter()


def instrument_httpx_propagation(session=None):
    """Create client spans for outbound HTTP and inject traceparent so gateway spans join the trace."""
    from opentelemetry import trace
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    provider = getattr(session, "_experimental_tracer_provider", None) or trace.get_tracer_provider()
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)


def _tag_rollout_impl(task_id: str, split: str, conversation_id: str, run_tag: str) -> str:
    return conversation_id


def trajectory_messages(result) -> list[dict]:
    """LangGraph result -> chat messages (user / assistant with tool_calls / tool), JSON-safe."""
    out = []
    for m in result["messages"]:
        t = getattr(m, "type", None)
        if t == "human":
            out.append({"role": "user", "content": content_text(m.content)})
        elif t == "ai":
            msg = {"role": "assistant", "content": content_text(m.content).strip()}
            if m.tool_calls:
                msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}}
                                     for c in m.tool_calls]
            out.append(msg)
        elif t == "tool":
            out.append({"role": "tool", "content": tool_text(m.content)})
    return out


def make_trajectory_recorder(session, app_name: str, app_version: str):
    """Puts a rollout's messages as JSON on a plain OTel span (attribute agt.trajectory), so
    trajectories are in AGENT_TRACE_TABLE even when the model is not called through the gateway.

    It is deliberately not a TruLens record: TruLens judges every record it finds, and this
    span has nothing to judge."""
    tracer = session._experimental_tracer_provider.get_tracer("agt.trajectory")

    def record(conversation_id: str, messages_json: str) -> None:
        with tracer.start_as_current_span("agt.trajectory") as span:
            span.set_attribute("ai.observability.app_name", app_name)
            span.set_attribute("ai.observability.app_version", app_version)
            span.set_attribute("agt.conversation_id", conversation_id)
            span.set_attribute("agt.trajectory", messages_json)

    return record


def make_tagger():
    """An instrumented no-op whose span records rollout identity inside the TruLens trace."""
    from trulens.core.otel.instrument import instrument

    return instrument(name="tag_rollout")(_tag_rollout_impl)


# --------------------------------------------------------------------------- feedbacks
# Judges used to select training runs. correctness is computed too, but only for reporting:
# production traffic has no answer key, so it never gates selection.
GPA_JUDGES = ["logical_consistency", "execution_efficiency", "tool_selection", "tool_calling"]


def make_feedbacks(snowpark_session, tasks_by_question: dict[str, dict]):
    from trulens.core import Feedback, Metric, Selector
    from trulens.providers.cortex import Cortex

    provider = Cortex(snowpark_session=snowpark_session, model_engine=JUDGE_MODEL)

    def correctness(question, answer: str) -> float:
        """Programmatic: final answer matches SQL ground truth. Reporting only."""
        t = tasks_by_question.get(question_text(question))
        return 1.0 if t and is_correct(answer, t["answer"]) else 0.0

    f_correct = Feedback(correctness, name="correctness").on_input().on_output()
    f_relevance = Feedback(provider.relevance_with_cot_reasons, name="answer_relevance").on_input().on_output()
    gpa = [Metric(implementation=getattr(provider, f"{name}_with_cot_reasons"), name=name,
                  selectors={"trace": Selector(trace_level=True)})
           for name in GPA_JUDGES]
    return [f_correct, f_relevance, *gpa]


def question_text(value) -> str:
    """TruGraph records the input message as a dict (or its repr); return the user text."""
    import ast

    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip()
    if isinstance(value, dict):
        if "messages" in value:
            value = value["messages"][-1]
        return str(value.get("content", "")).strip()
    return str(value).strip()


def last_user_text(graph_input) -> str:
    msgs = graph_input["messages"]
    m = msgs[-1]
    return m["content"] if isinstance(m, dict) else m.content


def final_text(result) -> str:
    return content_text(result["messages"][-1].content)


def content_text(content) -> str:
    """Anthropic messages return content blocks (TruLens records them as a JSON string);
    keep only the text blocks, dropping thinking/tool_use blocks. Not for tool results, whose
    JSON rows look like content blocks; use tool_text for those."""
    if isinstance(content, str) and content.lstrip().startswith("[{"):
        try:
            content = json.loads(content)
        except ValueError:
            return content
    if isinstance(content, list):
        return "".join(
            (b.get("text", "") if b.get("type", "text") == "text" else "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return content or ""
