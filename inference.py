#!/usr/bin/env python3
"""
inference.py — Baseline LLM agent for DevOps/FinOps OpenEnv.

Runs all 3 tasks sequentially using an LLM via the OpenAI client.
Prints baseline scores for each task and a summary table.

Required environment variables:
  API_BASE_URL  — LLM endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace API key (used as api_key for OpenAI client)

Optional:
  ENV_URL       — environment server URL (default: http://localhost:7860)
"""

import json
import os
import sys
import time
import traceback

import requests
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
REQUEST_TIMEOUT = 30
LLM_TIMEOUT = 60

# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert DevOps/FinOps engineer managing cloud infrastructure.
Your goal: minimize cost while keeping all services healthy (meeting latency SLAs).
You get scored 0.0-1.0 based on: cost reduction + zero SLA violations + no orphaned workloads.

## Instance types (cost per hour)
  nano:   1 CPU,  1 GB, $0.05/hr
  micro:  1 CPU,  2 GB, $0.10/hr
  small:  2 CPU,  4 GB, $0.20/hr
  medium: 4 CPU,  8 GB, $0.40/hr
  large:  8 CPU, 16 GB, $0.80/hr
  xlarge: 16 CPU, 32 GB, $1.60/hr

## Available actions (respond with exactly ONE JSON object per turn)
1. provision — spin up a new empty server
   {"action_type": "provision", "instance_type": "<tier>"}
2. terminate — destroy a server (DANGER: orphans workloads if not empty!)
   {"action_type": "terminate", "server_id": "<id>"}
3. resize — change server size in-place (keeps workloads assigned)
   {"action_type": "resize", "server_id": "<id>", "instance_type": "<tier>"}
4. migrate — move a workload to a different server
   {"action_type": "migrate", "workload_id": "<id>", "target_server_id": "<id>"}
5. noop — do nothing
   {"action_type": "noop"}

## Critical ordering rules (violating these tanks your score)
1. Fix SLA breaches FIRST — migrate overloaded workloads to less loaded servers
2. Assign orphaned/unassigned workloads IMMEDIATELY
3. Migrate ALL workloads off a server BEFORE terminating it (never orphan workloads)
4. Resize only when the new size still fits all assigned workloads' CPU+RAM needs
5. Terminate idle/empty servers to cut cost
6. Prefer resize over provision+migrate (fewer actions, less cost)

## Heuristics
- Pick the SMALLEST instance type that fits combined CPU+RAM of all workloads on it
- Critical workloads (is_critical=true) carry 2x penalty weight — prioritize their SLAs
- If budget is impossible to meet: still maximize cost reduction — partial improvement scores well
- Consolidating onto fewer servers is almost always better than spreading across many
- Avoid repeating the same action (causes penalty)
- Avoid 3+ consecutive noops (causes penalty)

## Response format
Brief reasoning (1-2 sentences), then a single JSON action. Example:
Srv-001 is xlarge but only needs micro capacity. Resizing to cut cost.
{"action_type": "resize", "server_id": "srv-001", "instance_type": "micro"}"""

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════


def build_user_prompt(obs: dict) -> str:
    """Convert an observation dict into a concise prompt for the LLM."""
    lines = []
    lines.append(f"Step {obs['step_number']}/{obs['max_steps']} | "
                 f"Cost: ${obs['total_cost_per_hour']:.2f}/hr | "
                 f"Budget: ${obs['budget_per_hour']:.2f}/hr | "
                 f"SLA violations: {obs['sla_violations']} | "
                 f"Unassigned: {obs['unassigned_workloads']}")
    lines.append(f"Message: {obs['message']}")

    lines.append("\n## Servers")
    for s in obs["servers"]:
        lines.append(
            f"  {s['server_id']} ({s['instance_type']}) "
            f"CPU:{s['cpu_cores']}c@{s['cpu_utilization']:.0%} "
            f"RAM:{s['ram_gb']}GB@{s['ram_utilization']:.0%} "
            f"${s['cost_per_hour']}/hr "
            f"workloads: {s['assigned_workloads']}"
        )

    lines.append("\n## Workloads")
    for w in obs["workloads"]:
        status = "DOWN" if w["assigned_server"] is None else (
            "BREACH" if w["current_latency_ms"] > w["sla_latency_ms"] else "OK"
        )
        crit = " [CRITICAL]" if w["is_critical"] else ""
        lines.append(
            f"  {w['workload_id']} \"{w['name']}\"{crit} "
            f"needs {w['required_cpu']}cpu/{w['required_ram']}GB "
            f"latency:{w['current_latency_ms']:.0f}ms/sla:{w['sla_latency_ms']:.0f}ms "
            f"-> {status} "
            f"on:{w['assigned_server'] or 'NONE'}"
        )

    lines.append("\nRespond with a single JSON action:")
    return "\n".join(lines)


def parse_action(text: str) -> dict:
    """Extract a JSON action from the model's response text."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in markdown code block
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Try to find first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    # Fallback: noop
    print(f"    WARNING: Could not parse action, falling back to noop")
    print(f"    Raw response: {text[:200]}")
    return {"action_type": "noop"}


def call_env(method: str, endpoint: str, body: dict = None) -> dict:
    """Make an HTTP request to the environment server."""
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=body or {}, timeout=REQUEST_TIMEOUT)
        else:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"    ENV ERROR: {e}")
        return {}


def call_llm(client: OpenAI, messages: list) -> str:
    """Call the LLM with retry logic."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                timeout=LLM_TIMEOUT,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"    LLM ERROR (attempt {attempt + 1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    print("    LLM FAILED after 3 attempts, using noop")
    return '{"action_type": "noop"}'


# ═══════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════════════


def run_task(client: OpenAI, task_id: str) -> float:
    """Run a single task and return the final score."""
    print(f"\n{'=' * 60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'=' * 60}")

    # Reset environment
    reset_data = call_env("POST", "/reset", {"task_id": task_id})
    if not reset_data:
        print(f"  Failed to reset task {task_id}")
        return 0.0

    obs = reset_data
    max_steps = obs.get("max_steps", 5)
    final_score = 0.0
    reward = {}

    print(f"  Budget: ${obs.get('budget_per_hour', 0):.2f}/hr | "
          f"Current cost: ${obs.get('total_cost_per_hour', 0):.2f}/hr | "
          f"Max steps: {max_steps}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(max_steps):
        if obs.get("done", False):
            break

        # Build prompt from observation
        user_prompt = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        # Keep conversation manageable (last 6 exchanges + system)
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        # Call LLM
        raw_response = call_llm(client, messages)
        messages.append({"role": "assistant", "content": raw_response})

        # Parse action
        action = parse_action(raw_response)
        print(f"  Step {step + 1}/{max_steps}: {action.get('action_type', '?')}", end="")
        if action.get("server_id"):
            print(f" server={action['server_id']}", end="")
        if action.get("instance_type"):
            print(f" type={action['instance_type']}", end="")
        if action.get("workload_id"):
            print(f" wl={action['workload_id']}", end="")
        if action.get("target_server_id"):
            print(f" target={action['target_server_id']}", end="")

        # Execute action
        step_result = call_env("POST", "/step", action)
        if not step_result:
            print(" -> ENV ERROR")
            continue

        obs = step_result.get("observation", obs)
        reward = step_result.get("reward", {})
        done = step_result.get("done", False)
        final_score = reward.get("score", 0.0)
        info_msg = step_result.get("info", {}).get("action_result", "")

        print(f" -> score={final_score:.4f} | {info_msg[:60]}")

        if done:
            break

    print(f"\n  FINAL SCORE: {final_score:.4f}")
    print(f"  Cost efficiency: {reward.get('cost_efficiency', 0):.4f}")
    print(f"  Performance:     {reward.get('performance_score', 0):.4f}")
    print(f"  Penalty:         {reward.get('penalty', 0):.4f}")
    return final_score


def main():
    print("=" * 60)
    print("DevOps/FinOps OpenEnv — Baseline Inference")
    print("=" * 60)

    # Validate environment variables
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL environment variable not set")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable not set")
        sys.exit(1)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set")
        sys.exit(1)

    print(f"  LLM endpoint: {API_BASE_URL}")
    print(f"  Model:        {MODEL_NAME}")
    print(f"  Env server:   {ENV_URL}")

    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    # Run all 3 tasks
    scores = {}
    start_time = time.time()

    for task_id in TASKS:
        task_start = time.time()
        try:
            score = run_task(client, task_id)
        except Exception as e:
            print(f"\n  TASK FAILED: {e}")
            traceback.print_exc()
            score = 0.0
        task_elapsed = time.time() - task_start
        scores[task_id] = score
        print(f"  Time: {task_elapsed:.1f}s")

    total_elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'=' * 60}")
    print("BASELINE SCORES SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Task':<12} {'Score':>8} {'Status':>10}")
    print(f"{'-' * 12} {'-' * 8} {'-' * 10}")
    for task_id in TASKS:
        s = scores[task_id]
        status = "PASS" if s > 0.0 else "FAIL"
        print(f"{task_id:<12} {s:>8.4f} {status:>10}")
    print(f"{'-' * 12} {'-' * 8} {'-' * 10}")
    avg = sum(scores.values()) / len(scores)
    print(f"{'AVERAGE':<12} {avg:>8.4f}")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
