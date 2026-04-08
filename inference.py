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

FATAL_IMPORT_ERROR = None

try:
    import requests
    from openai import OpenAI
except ImportError as ie:
    FATAL_IMPORT_ERROR = f"IMPORT_ERROR_{ie}".replace(" ", "_")
    # Will gracefully fail in main()

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or ""
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

## Actions (respond with exactly ONE JSON object per turn)
1. provision: {"action_type":"provision","instance_type":"<tier>"}
2. terminate: {"action_type":"terminate","server_id":"<id>"}
3. resize:    {"action_type":"resize","server_id":"<id>","instance_type":"<tier>"}
4. migrate:   {"action_type":"migrate","workload_id":"<id>","target_server_id":"<id>"}
5. noop:      {"action_type":"noop"}

## Response format
Respond ONLY with a single JSON action"""

# ═══════════════════════════════════════════════════════════════════════
# LOGGING & HELPERS (STRICT FORMATTING)
# ═══════════════════════════════════════════════════════════════════════

def format_safe_string(text: str) -> str:
    # Removes all spaces/newlines so key=value parser doesn't break
    if not text:
        return "null"
    return str(text).replace(" ", "_").replace("\n", "_").replace("\r", "")

def log_start(task: str, env: str, model: str):
    # e.g. [START] task=easy env=devops-finops model=Qwen...
    # Make sure task, env, model don't contain raw spaces if passing them
    t_safe = format_safe_string(task)
    e_safe = format_safe_string(env)
    m_safe = format_safe_string(model)
    print(f"[START] task={t_safe} env={e_safe} model={m_safe}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    # Action string MUST HAVE NO SPACES. 
    if isinstance(action, dict):
        action_str = json.dumps(action, separators=(',', ':'))
    else:
        # If it's a string, strip all spaces
        action_str = str(action).replace('\n', '').replace('\r', '').replace(' ', '')
        if not action_str:
            action_str = "noop"
            
    done_str = str(done).lower()
    error_safe = format_safe_string(error) if error else "null"
    
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_safe}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    success_str = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    if not rewards_str:
        rewards_str = "0.00"
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def build_user_prompt(obs: dict) -> str:
    # Keeping minimal prompt logic to ensure no hallucinated strings
    return f"Observation: {json.dumps(obs, separators=(',', ':'))}\nRespond with JSON action."

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
    print(f"    WARNING: Could not parse action\nRaw: {text[:50]}", file=sys.stderr)
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
        print(f"    ENV ERROR: {e}", file=sys.stderr)
        return {}


def call_llm(client, messages: list) -> str:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                temperature=0.0, max_tokens=512, timeout=LLM_TIMEOUT,
            )
            return response.choices[0].message.content or '{"action_type":"noop"}'
        except Exception as e:
            print(f"    LLM ERROR: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(1)
    return '{"action_type":"noop"}'


# ═══════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════════════

def run_task(client, task_id: str) -> float:
    # Always guarantee START logs first
    log_start(task=task_id, env="devops-finops-cloud-optimizer", model=MODEL_NAME)

    reset_data = call_env("POST", "/reset", {"task_id": task_id})
    if not reset_data:
        log_step(step=1, action="noop", reward=0.0, done=True, error="HTTP_CONNECTION_REFUSED")
        log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        return 0.0

    session_id = reset_data.get("session_id")
    obs = reset_data
    max_steps = obs.get("max_steps", 5)
    final_score = 0.0
    action_log = []
    rewards = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps_taken = 0

    for step_num in range(max_steps):
        steps_taken = step_num + 1
        if obs.get("done", False):
            break

        user_prompt = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_prompt})

        # Keep context window tight
        if len(messages) > 7:
            messages = [messages[0]] + messages[-6:]

        raw_response = call_llm(client, messages)
        messages.append({"role": "assistant", "content": raw_response})

        action = parse_action(raw_response)
        
        step_result = call_env("POST", "/step", action, {"session_id": session_id})
        error = None
        done = False
        reward_val = 0.0

        if not step_result:
            error = "ENV_STEP_FAILED"
        else:
            obs = step_result.get("observation", obs)
            reward_dict = step_result.get("reward", {})
            done = step_result.get("done", False)
            final_score = reward_dict.get("score", 0.0)
            reward_val = final_score
            info_msg = step_result.get("info", {}).get("action_result", "")
            action_log.append((action, info_msg))

        rewards.append(reward_val)
        
        # Log step with raw_response, but the log_step helper strips all spaces
        log_step(step=steps_taken, action=action, reward=reward_val, done=done, error=error)

        if done:
            break

    success = final_score > 0.0  # Any partial credit is a minimal success
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score


def main():
    if FATAL_IMPORT_ERROR:
        print(f"[START] task=fatal_error env=devops-finops model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=1 action=noop reward=0.00 done=true error={FATAL_IMPORT_ERROR}", flush=True)
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
        sys.exit(0)

    try:
        if "OpenAI" in globals():
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")
        else:
            raise Exception("OpenAI not loaded")
    except Exception as fatal_e:
        err_str = format_safe_string(f"CLIENT_ERROR_{fatal_e}")
        print(f"[START] task=error env=devops-finops model={MODEL_NAME}", flush=True)
        print(f"[STEP] step=1 action=noop reward=0.00 done=true error={err_str}", flush=True)
        print(f"[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
        sys.exit(0)

    scores = {}
    for task_id in TASKS:
        try:
            score = run_task(client, task_id)
        except Exception as e:
            err_str = format_safe_string(f"UNHANDLED_{e}")
            print(f"[START] task={task_id} env=devops-finops model={MODEL_NAME}", flush=True)
            print(f"[STEP] step=1 action=noop reward=0.00 done=true error={err_str}", flush=True)
            print(f"[END] success=false steps=1 score=0.00 rewards=0.00", flush=True)
            score = 0.0
        scores[task_id] = score


if __name__ == "__main__":
    main()
