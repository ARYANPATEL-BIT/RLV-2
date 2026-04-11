#!/usr/bin/env python3
"""
inference.py — Baseline LLM agent for DevOps/FinOps OpenEnv v2.0.

Runs all 3 tasks (+ optional random) sequentially using an LLM.
Handles session management, spot evictions, cascading failures.

Required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
Optional: ENV_URL (default: http://localhost:7860)
"""

# Force unbuffered stdout BEFORE anything else
import os as _os
_os.environ["PYTHONUNBUFFERED"] = "1"

import json
import math
import os
import sys
import time
import traceback

# Ensure stdout is truly unbuffered (belt-and-suspenders)
try:
    sys.stdout.reconfigure(line_buffering=False, write_through=True)
except Exception:
    pass

try:
    import requests
except ImportError:
    requests = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ENV_URL resolution order:
# 1. Explicit ENV_URL env var
# 2. Constructed from SPACE_ID (HuggingFace Spaces pattern)
# 3. Default to localhost:7860
def _resolve_env_url() -> str:
    explicit = os.environ.get("ENV_URL", "")
    if explicit:
        return explicit.rstrip("/")
    space_id = os.environ.get("SPACE_ID", "")
    if space_id and "/" in space_id:
        user, repo = space_id.split("/", 1)
        return f"https://{user}-{repo}.hf.space"
    return "http://localhost:7860"

ENV_URL = _resolve_env_url()

TASKS = ["easy", "medium", "hard"]
REQUEST_TIMEOUT = 30
LLM_TIMEOUT = 60
ENV_BOOT_RETRIES = 3
ENV_BOOT_DELAY = 2

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
2. terminate: {"action_type": "terminate", "server_id": "<id>"}
3. resize:    {"action_type": "resize", "server_id": "<id>", "instance_type": "<tier>"}
4. migrate:   {"action_type": "migrate", "workload_id": "<id>", "target_server_id": "<id>"}
5. noop:      {"action_type": "noop"}

## Critical rules
1. Fix SLA breaches FIRST — check ALL 4 dimensions: CPU, RAM, disk IOPS, network
2. Assign orphaned workloads IMMEDIATELY
3. Migrate ALL workloads off a server BEFORE terminating it
4. Check workload DEPENDENCIES — if dependency is down/breaching, dependent cascades
5. Spot instances evict at steps 3/7/12
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


def _safe_float(val, default=0.0):
    """Convert to float safely, replacing NaN/Inf with default."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_json(obj):
    """json.dumps that never raises."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return str(obj)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    reward = _safe_float(reward)
    error_part = f" error={error}" if error else ""
    print(f"[STEP] step={step} reward={reward} done={done}{error_part}", flush=True)


def log_end(task, success, steps, score):
    score = _safe_float(score)
    print(f"[END] task={task} success={success} score={score} steps={steps}", flush=True)


def build_user_prompt(obs):
    """Build a user prompt from the observation dict. Fully defensive with .get()."""
    try:
        lines = []
        step_num = obs.get('step_number', '?')
        max_steps = obs.get('max_steps', '?')
        cost = _safe_float(obs.get('total_cost_per_hour', 0))
        budget = _safe_float(obs.get('budget_per_hour', 0))
        sla_v = obs.get('sla_violations', 0)
        unassigned = obs.get('unassigned_workloads', 0)

        lines.append(f"Step {step_num}/{max_steps} | "
                     f"Cost: ${cost:.2f}/hr | "
                     f"Budget: ${budget:.2f}/hr | "
                     f"SLA violations: {sla_v} | "
                     f"Unassigned: {unassigned}")
        if obs.get("spot_eviction_occurred"):
            lines.append("SPOT EVICTION occurred this step!")
        if obs.get("traffic_multiplier_active"):
            lines.append("Traffic spikes active (demand fluctuates each step)")
        lines.append(f"Message: {obs.get('message', 'N/A')}")

        lines.append("\n## Servers")
        for s in obs.get("servers", []):
            badges = ""
            if s.get("is_spot"):
                badges += " [SPOT]"
            if s.get("is_overloaded"):
                badges += " [OVERLOADED]"
            lines.append(
                f"  {s.get('server_id', '?')} ({s.get('instance_type', '?')}) "
                f"CPU:{s.get('cpu_cores', 0)}c@{_safe_float(s.get('cpu_utilization', 0)):.0%} "
                f"RAM:{s.get('ram_gb', 0)}GB@{_safe_float(s.get('ram_utilization', 0)):.0%} "
                f"Disk:{s.get('disk_iops', 0)}iops@{_safe_float(s.get('disk_utilization', 0)):.0%} "
                f"Net:{s.get('network_gbps', 0)}Gbps@{_safe_float(s.get('network_utilization', 0)):.0%} "
                f"${s.get('cost_per_hour', 0)}/hr "
                f"wl: {s.get('assigned_workloads', [])}{badges}"
            )

        lines.append("\n## Workloads")
        for w in obs.get("workloads", []):
            assigned = w.get("assigned_server")
            cur_lat = _safe_float(w.get("current_latency_ms", 0))
            sla_lat = _safe_float(w.get("sla_latency_ms", 1))
            if assigned is None:
                status = "DOWN"
            elif cur_lat > sla_lat:
                status = "BREACH"
            else:
                status = "OK"
            crit = " [CRITICAL]" if w.get("is_critical") else ""
            deps = f" depends->{w.get('dependencies', [])}" if w.get("dependencies") else ""
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
        return f"Observation (raw): {str(obs)[:2000]}\n\nRespond with a single JSON action:"


def parse_action(text):
    """Parse an LLM response into an action dict. Never raises."""
    if not text:
        return {"action_type": "noop"}
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
    return {"action_type": "noop"}


def call_env(method, endpoint, body=None, params=None):
    """Call the environment HTTP API. Returns dict or {} on failure."""
    if requests is None:
        return {}
    url = f"{ENV_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=body or {}, params=params, timeout=REQUEST_TIMEOUT)
        else:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    ENV ERROR [{method} {endpoint}]: {e}", flush=True)
        return {}


def wait_for_env():
    """Wait for the env container to be reachable, with retries."""
    if requests is None:
        print("    WARN: requests not installed, cannot check env", flush=True)
        return False
    for attempt in range(ENV_BOOT_RETRIES):
        try:
            resp = requests.get(f"{ENV_URL}/health", timeout=10)
            if resp.status_code == 200:
                print(f"    ENV reachable on attempt {attempt + 1}", flush=True)
                return True
        except Exception:
            pass
        print(f"    ENV not ready, retry {attempt + 1}/{ENV_BOOT_RETRIES}...", flush=True)
        time.sleep(ENV_BOOT_DELAY)
    print("    ENV unreachable after retries", flush=True)
    return False


def call_llm(client, messages):
    """Call the LLM. Returns response text or noop JSON on failure."""
    if client is None:
        return '{"action_type": "noop"}'
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=0.0, max_tokens=512, timeout=LLM_TIMEOUT,
            )
            return response.choices[0].message.content or '{"action_type": "noop"}'
        except Exception as e:
            print(f"    LLM ERROR (attempt {attempt + 1}/3): {e}", flush=True)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return '{"action_type": "noop"}'


# ═══════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════════════


def run_task(client, task_id):
    """Run a single task. ALWAYS prints [START] and [END]."""
    final_score = 0.0
    steps_taken = 0
    rewards = []

    # ALWAYS print [START] first, before anything that could fail
    log_start(task=task_id, env="devops-finops-cloud-optimizer", model=MODEL_NAME or "unknown")

    try:
        reset_data = call_env("POST", "/reset", {"task_id": task_id})

        if not reset_data:
            print(f"    RESET returned empty for task {task_id}", flush=True)
            # Still do 1 noop step so we have a [STEP]
            log_step(step=1, action='{"action_type": "noop"}', reward=0.0, done=True, error="ENV unreachable")
            rewards.append(0.0)
            steps_taken = 1
            log_end(task=task_id, success=False, steps=steps_taken, score=0.0)
            return 0.0

        session_id = reset_data.get("session_id")
        if not session_id:
            print(f"    RESET did not return session_id for task {task_id}", flush=True)
            log_step(step=1, action='{"action_type": "noop"}', reward=0.0, done=True, error="No session_id")
            rewards.append(0.0)
            steps_taken = 1
            log_end(task=task_id, success=False, steps=steps_taken, score=0.0)
            return 0.0

        obs = reset_data
        max_steps = obs.get("max_steps", 5)
        action_log = []

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step_num in range(max_steps):
            steps_taken = step_num + 1
            if obs.get("done", False):
                break

            user_prompt = build_user_prompt(obs)
            if action_log:
                user_prompt += f"\n\n## Action History\nYou have taken {len(action_log)} actions so far:\n"
                for past_step, (past_a, past_m) in enumerate(action_log):
                    at = past_a.get('action_type', '?') if isinstance(past_a, dict) else '?'
                    user_prompt += f"  Step {past_step+1}: {at} -> {str(past_m)[:70]}\n"
            messages.append({"role": "user", "content": user_prompt})

            # Keep conversation manageable
            if len(messages) > 13:
                messages = [messages[0]] + messages[-12:]

            raw_response = call_llm(client, messages)
            messages.append({"role": "assistant", "content": raw_response})

            action = parse_action(raw_response)

            step_result = call_env("POST", "/step", action, {"session_id": session_id})

            error = None
            done = False
            reward_val = 0.0

            if not step_result:
                error = "ENV ERROR"
            else:
                obs = step_result.get("observation", obs)
                reward_dict = step_result.get("reward", {})
                if isinstance(reward_dict, dict):
                    final_score = _safe_float(reward_dict.get("score", 0.0))
                else:
                    final_score = 0.0
                reward_val = final_score
                done = step_result.get("done", False)
                info = step_result.get("info", {})
                info_msg = info.get("action_result", "") if isinstance(info, dict) else str(info)
                action_log.append((action, info_msg))

            rewards.append(reward_val)
            log_step(step=steps_taken, action=raw_response, reward=reward_val, done=done, error=error)

            if done:
                break

    except Exception as e:
        print(f"    TASK EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        if steps_taken == 0:
            steps_taken = 1
            rewards.append(0.0)
            log_step(step=1, action='{"action_type": "noop"}', reward=0.0, done=True, error=str(e))

    # ALWAYS print [END] — guaranteed
    success = final_score > 0.0
    log_end(task=task_id, success=success, steps=steps_taken, score=final_score)
    return final_score


def main():
    # Immediately print something to stdout so the validator knows we're alive
    sys.stdout.write("")
    sys.stdout.flush()

    print("="*60, flush=True)
    print("DevOps/FinOps OpenEnv — Baseline Inference Agent v2.0", flush=True)
    print("="*60, flush=True)
    print(f"[CONFIG] ENV_URL={ENV_URL}", flush=True)
    print(f"[CONFIG] MODEL={MODEL_NAME or 'unset'}", flush=True)
    print(f"[CONFIG] API_BASE={API_BASE_URL[:50] if API_BASE_URL else 'unset'}", flush=True)
    print(f"[CONFIG] TASKS={TASKS}", flush=True)

    # Create LLM client (or None if not available)
    client = None
    if OpenAI is not None and API_BASE_URL and HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception as e:
            print(f"[WARN] Failed to create OpenAI client: {e}", flush=True)
    else:
        missing = []
        if OpenAI is None:
            missing.append("openai package")
        if not API_BASE_URL:
            missing.append("API_BASE_URL")
        if not HF_TOKEN:
            missing.append("HF_TOKEN")
        print(f"[WARN] LLM unavailable (missing: {', '.join(missing)}), using noop fallback", flush=True)

    # Wait for env container to boot (do this AFTER LLM client setup to minimize
    # delay before structured output)
    env_ready = wait_for_env()
    if not env_ready:
        print("[WARN] Env not reachable, will still attempt tasks", flush=True)

    scores = {}
    for task_id in TASKS:
        try:
            score = run_task(client, task_id)
        except Exception as e:
            print(f"[ERROR] Task '{task_id}' crashed: {e}", flush=True)
            traceback.print_exc()
            score = 0.0
            # Emit structured output even for crashed tasks
            log_start(task=task_id, env="devops-finops-cloud-optimizer", model=MODEL_NAME or "unknown")
            log_step(step=1, action='{"action_type": "noop"}', reward=0.0, done=True, error=str(e))
            log_end(task=task_id, success=False, steps=1, score=0.0)
        scores[task_id] = score

    print(f"[RESULTS] {_safe_json(scores)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        print(f"[FATAL] {e}", flush=True)
        traceback.print_exc()
        # Still emit minimal structured output so validator finds markers
        for t in TASKS:
            log_start(task=t, env="devops-finops-cloud-optimizer", model="unknown")
            log_step(step=1, action='{"action_type": "noop"}', reward=0.0, done=True, error=str(e))
            log_end(task=t, success=False, steps=1, score=0.0)
