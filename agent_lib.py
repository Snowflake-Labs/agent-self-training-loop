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
TEACHER_MODEL = "claude-opus-5"  # frontier teacher; the student is Qwen3-8B post-trained on its best traces
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
SYSTEM_PROMPTS = {"basic": SYSTEM_PROMPT, "hard": SYSTEM_PROMPT_HARD}
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


def suite(name: str):
    """(db, tasks, system_prompt) for a task suite."""
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
def build_agent(conn: sqlite3.Connection, api_key: str, conversation_id: str,
                model: str = TEACHER_MODEL, base_url: str | None = None, temperature: float = 0.7,
                system_prompt: str = SYSTEM_PROMPT):
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    @tool
    def sql_query(query: str) -> str:
        """Run a read-only SQLite query and return up to 50 rows as JSON."""
        if not re.match(r"^\s*(select|with)\b", query, re.IGNORECASE):
            return "ERROR: only SELECT queries are allowed"
        try:
            cur = conn.execute(query)
            cols = [d[0] for d in cur.description]
            return json.dumps([dict(zip(cols, r)) for r in cur.fetchmany(50)])
        except Exception as e:  # surfaced to the model
            return f"ERROR: {e}"

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression with + - * / ( ) and decimals."""
        if not re.fullmatch(r"[0-9.+\-*/() eE]+", expression):
            return "ERROR: unsupported characters"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - charset restricted above
        except Exception as e:
            return f"ERROR: {e}"

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
            temperature=temperature, max_retries=2, timeout=120, max_completion_tokens=1024,
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


def make_tagger():
    """An instrumented no-op whose span records rollout identity inside the TruLens trace."""
    from trulens.core.otel.instrument import instrument

    return instrument(name="tag_rollout")(_tag_rollout_impl)


# --------------------------------------------------------------------------- feedbacks
def make_feedbacks(snowpark_session, tasks_by_question: dict[str, dict]):
    from trulens.core import Feedback
    from trulens.providers.cortex import Cortex

    provider = Cortex(snowpark_session=snowpark_session, model_engine=JUDGE_MODEL)

    def correctness(question, answer: str) -> float:
        """Programmatic: final answer matches SQL ground truth."""
        t = tasks_by_question.get(question_text(question))
        return 1.0 if t and is_correct(answer, t["answer"]) else 0.0

    f_correct = Feedback(correctness, name="correctness").on_input().on_output()
    f_relevance = Feedback(provider.relevance_with_cot_reasons, name="answer_relevance").on_input().on_output()
    return [f_correct, f_relevance]


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
    keep only the text blocks, dropping thinking/tool_use blocks."""
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
