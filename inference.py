#!/usr/bin/env python3
"""
inference.py — Baseline LLM agent for DevOps/FinOps OpenEnv v2.0.

Runs all 3 tasks (+ optional random) sequentially using an LLM.
Handles session management, spot evictions, cascading failures.

Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
Optional: ENV_URL (default: http://localhost:7860)
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
# SYSTEM PROMPT — updated for v2.0
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert DevOps/FinOps engineer managing cloud infrastructure.
Your goal: minimize cost while keeping all services healthy (meeting latency SLAs).
Score 0.0-1.0: cost_efficiency×0.40 + performance×0.35 - penalty×0.25.

## Instance types (cost/hr)
  nano:        1cpu,  1GB,  1k IOPS, 0.5Gbps, $0.05
  micro:       1cpu,  2GB,  2k IOPS, 0.5Gbps, $0.10
  small:       2cpu,  4GB,  3k IOPS, 1.0Gbps, $0.20
  medium:      4cpu,  8GB,  5k IOPS, 2.0Gbps, $0.40
  large:       8cpu, 16GB, 10k IOPS, 5.0Gbps, $0.80
  xlarge:     16cpu, 32GB, 20k IOPS, 10Gbps,  $1.60
  compute-opt: 8cpu,  8GB,  5k IOPS, 5.0Gbps, $0.60  (high CPU, low RAM)
  memory-opt:  4cpu, 32GB, 10k IOPS, 2.0Gbps, $0.70  (high RAM)
  storage-opt: 4cpu,  8GB, 30k IOPS, 2.0Gbps, $0.55  (high disk)
  spot-medium: 4cpu,  8GB,  5k IOPS, 2.0Gbps, $0.12  (CHEAP but can be EVICTED!)
  spot-large:  8cpu, 16GB, 10k IOPS, 5.0Gbps, $0.24  (CHEAP but can be EVICTED!)

## Actions (respond with exactly ONE JSON object per turn)
1. provision: {"action_type": "provision", "instance_type": "<tier>"}
2. terminate: {"action_type": "terminate", "server_id": "<id>"}  ← DANGER: orphans workloads!
3. resize:    {"action_type": "resize", "server_id": "<id>", "instance_type": "<tier>"}
4. migrate:   {"action_type": "migrate", "workload_id": "<id>", "target_server_id": "<id>"}
5. noop:      {"action_type": "noop"}

## Critical rules
1. Fix SLA breaches FIRST — check ALL 4 dimensions: CPU, RAM, disk IOPS, network
2. Assign orphaned workloads IMMEDIATELY
3. Migrate ALL workloads off a server BEFORE terminating it
4. Check workload DEPENDENCIES — if dependency is down/breaching, dependent cascades
5. Spot instances evict at steps 3/7/12 — don't rely on them for critical workloads
6. Resize keeps workloads — verify new size fits all workloads on it
7. Destructive termination = 0.50 penalty (VERY expensive with 0.25 weight)
8. Prefer resize over provision+migrate

## Heuristics
- Pick SMALLEST instance fitting combined CPU+RAM+disk+net of all workloads
- Critical workloads carry 2x penalty weight — prioritize their SLAs
- Consolidate onto fewer servers when possible
- Never use spot instances for critical workloads
- Context-switch overhead: +5% utilization per extra workload on same server

## Response format
Brief reasoning (1-2 sentences), then a single JSON action:
Srv-001 is xlarge but only needs small capacity. Resizing to cut cost.
{"action_type": "resize", "server_id": "srv-001", "instance_type": "small"}"""

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════


def build_user_prompt(obs: dict) -> str:
    lines = []
    lines.append(f"Step {obs['step_number']}/{obs['max_steps']} | "
                 f"Cost: ${obs['total_cost_per_hour']:.2f}/hr | "
                 f"Budget: ${obs['budget_per_hour']:.2f}/hr | "
                 f"SLA violations: {obs['sla_violations']} | "
                 f"Unassigned: {obs['unassigned_workloads']}")
    if obs.get("spot_eviction_occurred"):
        lines.append("⚠️ SPOT EVICTION occurred this step!")
    if obs.get("traffic_multiplier_active"):
        lines.append("📈 Traffic spikes active (demand fluctuates each step)")
    lines.append(f"Message: {obs['message']}")

    lines.append("\n## Servers")
    for s in obs["servers"]:
        badges = ""
        if s.get("is_spot"):
            badges += " [SPOT]"
        if s.get("is_overloaded"):
            badges += " [OVERLOADED]"
        lines.append(
            f"  {s['server_id']} ({s['instance_type']}) "
            f"CPU:{s['cpu_cores']}c@{s['cpu_utilization']:.0%} "
            f"RAM:{s['ram_gb']}GB@{s['ram_utilization']:.0%} "
            f"Disk:{s.get('disk_iops',0)}iops@{s.get('disk_utilization',0):.0%} "
            f"Net:{s.get('network_gbps',0)}Gbps@{s.get('network_utilization',0):.0%} "
            f"${s['cost_per_hour']}/hr "
            f"wl: {s['assigned_workloads']}{badges}"
        )

    lines.append("\n## Workloads")
    for w in obs["workloads"]:
        status = "DOWN" if w["assigned_server"] is None else (
            "BREACH" if w["current_latency_ms"] > w["sla_latency_ms"] else "OK"
        )
        crit = " [CRITICAL]" if w["is_critical"] else ""
        deps = f" depends→{w['dependencies']}" if w.get("dependencies") else ""
        lines.append(
            f"  {w['workload_id']} \"{w['name']}\"{crit} "
            f"needs {w['required_cpu']}cpu/{w['required_ram']}GB/"
            f"{w.get('required_disk_iops',0)}iops/{w.get('required_net_gbps',0)}Gbps "
            f"latency:{w['current_latency_ms']:.0f}ms/sla:{w['sla_latency_ms']:.0f}ms "
            f"-> {status} on:{w['assigned_server'] or 'NONE'}{deps}"
        )

    lines.append("\nRespond with a single JSON action:")
    return "\n".join(lines)


def parse_action(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass
    print(f"    WARNING: Could not parse action, falling back to noop")
    print(f"    Raw response: {text[:200]}")
    return {"action_type": "noop"}


def call_env(method: str, endpoint: str, body: dict = None, params: dict = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=body or {}, params=params, timeout=REQUEST_TIMEOUT)
        else:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"    ENV ERROR: {e}")
        return {}


def call_llm(client: OpenAI, messages: list) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=0.0, max_tokens=512, timeout=LLM_TIMEOUT,
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
    print(f"\n{'=' * 60}")
    print(f"TASK: {task_id.upper()}")
    print(f"{'=' * 60}")

    reset_data = call_env("POST", "/reset", {"task_id": task_id})
    if not reset_data:
        print(f"  Failed to reset task {task_id}")
        return 0.0

    session_id = reset_data.get("session_id")
    obs = reset_data
    max_steps = obs.get("max_steps", 5)
    final_score = 0.0
    reward = {}
    action_log = []

    print(f"  Session: {session_id}")
    print(f"  Budget: ${obs.get('budget_per_hour', 0):.2f}/hr | "
          f"Current cost: ${obs.get('total_cost_per_hour', 0):.2f}/hr | "
          f"Max steps: {max_steps}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step_num in range(max_steps):
        if obs.get("done", False):
            break

        user_prompt = build_user_prompt(obs)
        if action_log:
            user_prompt += f"\n\n## Action History\nYou have taken {len(action_log)} actions so far:\n"
            for past_step, (past_a, past_m) in enumerate(action_log):
                user_prompt += f"  Step {past_step+1}: {past_a.get('action_type', '?')} -> {past_m[:70]}\n"
        messages.append({"role": "user", "content": user_prompt})

        # Keep conversation manageable (last 6 exchanges + system)
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        raw_response = call_llm(client, messages)
        messages.append({"role": "assistant", "content": raw_response})

        action = parse_action(raw_response)
        print(f"  Step {step_num + 1}/{max_steps}: {action.get('action_type', '?')}", end="")
        for k in ["server_id", "instance_type", "workload_id", "target_server_id"]:
            if action.get(k):
                print(f" {k}={action[k]}", end="")

        step_result = call_env("POST", "/step", action, {"session_id": session_id})
        if not step_result:
            print(" -> ENV ERROR")
            continue

        obs = step_result.get("observation", obs)
        reward = step_result.get("reward", {})
        done = step_result.get("done", False)
        final_score = reward.get("score", 0.0)
        info_msg = step_result.get("info", {}).get("action_result", "")
        action_log.append((action, info_msg))

        print(f" -> score={final_score:.4f} | {info_msg[:70]}")

        if done:
            break

    print(f"\n  FINAL SCORE: {final_score:.4f}")
    print(f"  Cost efficiency: {reward.get('cost_efficiency', 0):.4f}")
    print(f"  Performance:     {reward.get('performance_score', 0):.4f}")
    print(f"  Penalty:         {reward.get('penalty', 0):.4f}")
    return final_score


def main():
    print("=" * 60)
    print("DevOps/FinOps OpenEnv v2.0 — Baseline Inference")
    print("=" * 60)

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

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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
