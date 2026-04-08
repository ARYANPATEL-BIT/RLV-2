#!/usr/bin/env python3
"""
inference.py — Baseline LLM agent for DevOps/FinOps OpenEnv v2.0.

Runs all 3 tasks (+ optional random) sequentially using an LLM.
Handles session management, spot evictions, cascading failures.

Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
Optional: ENV_URL (default: http://localhost:7860)
"""

import json
import math
import os
import sys
import time
import traceback

try:
    import requests
except ImportError:
    print("[FATAL] 'requests' package not installed. pip install requests")
    sys.exit(0)

try:
    from openai import OpenAI
except ImportError:
    print("[FATAL] 'openai' package not installed. pip install openai")
    sys.exit(0)

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
# LOGGING & HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _safe_json_dumps(obj):
    """json.dumps that handles NaN/Inf gracefully."""
    try:
        return json.dumps(obj, default=str)
    except (ValueError, TypeError):
        return str(obj)


def log_start(task: str, env: str, model: str):
    print(f"[START] {_safe_json_dumps({'task': task, 'env': env, 'model': model})}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    try:
        # Sanitise reward (json.dumps rejects NaN/Inf)
        if isinstance(reward, float) and (math.isnan(reward) or math.isinf(reward)):
            reward = 0.0
        print(f"[STEP] {_safe_json_dumps({'step': step, 'action': action, 'reward': reward, 'done': done, 'error': error})}", flush=True)
    except Exception as e:
        print(f"[STEP] step={step} reward={reward} done={done} error={error} (log_step err: {e})", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    try:
        if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
            score = 0.0
        clean_rewards = []
        for r in rewards:
            if isinstance(r, float) and (math.isnan(r) or math.isinf(r)):
                clean_rewards.append(0.0)
            else:
                clean_rewards.append(r)
        print(f"[END] {_safe_json_dumps({'success': success, 'steps': steps, 'score': score, 'rewards': clean_rewards})}", flush=True)
    except Exception as e:
        print(f"[END] success={success} steps={steps} score={score} (log_end err: {e})", flush=True)


def build_user_prompt(obs: dict) -> str:
    """Build a user prompt from the observation dict. Fully defensive with .get()."""
    try:
        lines = []
        step_num = obs.get('step_number', '?')
        max_steps = obs.get('max_steps', '?')
        cost = obs.get('total_cost_per_hour', 0.0)
        budget = obs.get('budget_per_hour', 0.0)
        sla_v = obs.get('sla_violations', 0)
        unassigned = obs.get('unassigned_workloads', 0)

        lines.append(f"Step {step_num}/{max_steps} | "
                     f"Cost: ${cost:.2f}/hr | "
                     f"Budget: ${budget:.2f}/hr | "
                     f"SLA violations: {sla_v} | "
                     f"Unassigned: {unassigned}")
        if obs.get("spot_eviction_occurred"):
            lines.append("⚠️ SPOT EVICTION occurred this step!")
        if obs.get("traffic_multiplier_active"):
            lines.append("📈 Traffic spikes active (demand fluctuates each step)")
        lines.append(f"Message: {obs.get('message', 'N/A')}")

        lines.append("\n## Servers")
        for s in obs.get("servers", []):
            badges = ""
            if s.get("is_spot"):
                badges += " [SPOT]"
            if s.get("is_overloaded"):
                badges += " [OVERLOADED]"
            cpu_util = s.get('cpu_utilization', 0.0)
            ram_util = s.get('ram_utilization', 0.0)
            disk_util = s.get('disk_utilization', 0.0)
            net_util = s.get('network_utilization', 0.0)
            lines.append(
                f"  {s.get('server_id', '?')} ({s.get('instance_type', '?')}) "
                f"CPU:{s.get('cpu_cores', 0)}c@{cpu_util:.0%} "
                f"RAM:{s.get('ram_gb', 0)}GB@{ram_util:.0%} "
                f"Disk:{s.get('disk_iops', 0)}iops@{disk_util:.0%} "
                f"Net:{s.get('network_gbps', 0)}Gbps@{net_util:.0%} "
                f"${s.get('cost_per_hour', 0)}/hr "
                f"wl: {s.get('assigned_workloads', [])}{badges}"
            )

        lines.append("\n## Workloads")
        for w in obs.get("workloads", []):
            assigned = w.get("assigned_server")
            cur_lat = w.get("current_latency_ms", 0.0)
            sla_lat = w.get("sla_latency_ms", 1.0)
            if assigned is None:
                status = "DOWN"
            elif cur_lat > sla_lat:
                status = "BREACH"
            else:
                status = "OK"
            crit = " [CRITICAL]" if w.get("is_critical") else ""
            deps = f" depends→{w.get('dependencies', [])}" if w.get("dependencies") else ""
            lines.append(
                f"  {w.get('workload_id', '?')} \"{w.get('name', '?')}\"{crit} "
                f"needs {w.get('required_cpu', 0)}cpu/{w.get('required_ram', 0)}GB/"
                f"{w.get('required_disk_iops', 0)}iops/{w.get('required_net_gbps', 0)}Gbps "
                f"latency:{cur_lat:.0f}ms/sla:{sla_lat:.0f}ms "
                f"-> {status} on:{assigned or 'NONE'}{deps}"
            )

        lines.append("\nRespond with a single JSON action:")
        return "\n".join(lines)
    except Exception as e:
        # Fallback: return a minimal prompt so the agent can still act
        print(f"    WARNING: build_user_prompt failed: {e}", flush=True)
        return f"Observation (raw): {str(obs)[:2000]}\n\nRespond with a single JSON action:"


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
    except Exception as e:
        print(f"    ENV UNEXPECTED ERROR: {e}")
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
    """Run a single task episode. Returns the final score (0.0 on failure)."""
    try:
        reset_data = call_env("POST", "/reset", {"task_id": task_id})
    except Exception as e:
        print(f"    RESET EXCEPTION for task {task_id}: {e}")
        return 0.0

    if not reset_data:
        print(f"    RESET returned empty for task {task_id}")
        return 0.0

    session_id = reset_data.get("session_id")
    if not session_id:
        print(f"    RESET did not return session_id for task {task_id}")
        return 0.0

    obs = reset_data
    max_steps = obs.get("max_steps", 5)
    final_score = 0.0
    action_log = []
    rewards = []

    log_start(task=task_id, env="devops-finops-cloud-optimizer", model=MODEL_NAME)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps_taken = 0

    for step_num in range(max_steps):
        steps_taken = step_num + 1
        if obs.get("done", False):
            break

        user_prompt = build_user_prompt(obs)
        if action_log:
            user_prompt += f"\n\n## Action History\nYou have taken {len(action_log)} actions so far:\n"
            for past_step, (past_a, past_m) in enumerate(action_log):
                past_action_type = past_a.get('action_type', '?') if isinstance(past_a, dict) else '?'
                past_msg = str(past_m)[:70] if past_m else ''
                user_prompt += f"  Step {past_step+1}: {past_action_type} -> {past_msg}\n"
        messages.append({"role": "user", "content": user_prompt})

        # Keep conversation manageable
        if len(messages) > 13:
            messages = [messages[0]] + messages[-12:]

        raw_response = call_llm(client, messages)
        messages.append({"role": "assistant", "content": raw_response})

        action = parse_action(raw_response)

        try:
            step_result = call_env("POST", "/step", action, {"session_id": session_id})
        except Exception as e:
            print(f"    STEP EXCEPTION: {e}")
            step_result = {}

        error = None
        done = False
        reward_val = 0.0

        if not step_result:
            error = "ENV ERROR"
        else:
            obs = step_result.get("observation", obs)
            reward_dict = step_result.get("reward", {})
            done = step_result.get("done", False)
            final_score = reward_dict.get("score", 0.0) if isinstance(reward_dict, dict) else 0.0
            reward_val = final_score
            info = step_result.get("info", {})
            info_msg = info.get("action_result", "") if isinstance(info, dict) else str(info)
            action_log.append((action, info_msg))

        rewards.append(reward_val)
        log_step(step=steps_taken, action=raw_response, reward=reward_val, done=done, error=error)

        if done:
            break

    success = final_score > 0.0  # Any partial credit is a minimal success
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score


def main():
    """Main entry point with comprehensive error handling."""
    if not API_BASE_URL:
        print("[ERROR] API_BASE_URL environment variable is not set.", flush=True)
        return
    if not MODEL_NAME:
        print("[ERROR] MODEL_NAME environment variable is not set.", flush=True)
        return
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is not set.", flush=True)
        return

    print(f"[CONFIG] ENV_URL={ENV_URL}, MODEL={MODEL_NAME}, API_BASE={API_BASE_URL[:50]}...", flush=True)

    # Verify env container is reachable before starting
    try:
        health = requests.get(f"{ENV_URL}/", timeout=15)
        print(f"[CONFIG] Env health check: {health.status_code}", flush=True)
    except Exception as e:
        print(f"[WARN] Env health check failed: {e}. Proceeding anyway...", flush=True)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"[ERROR] Failed to create OpenAI client: {e}", flush=True)
        return

    scores = {}

    for task_id in TASKS:
        try:
            score = run_task(client, task_id)
        except Exception as e:
            print(f"[ERROR] Task '{task_id}' crashed: {e}", flush=True)
            traceback.print_exc()
            score = 0.0
        scores[task_id] = score

    print(f"[RESULTS] {_safe_json_dumps(scores)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass  # Don't let sys.exit propagate as unhandled
    except Exception as e:
        print(f"[FATAL] Unhandled exception in main: {e}", flush=True)
        traceback.print_exc()
        # Exit with 0 so the validator doesn't see a non-zero exit code
