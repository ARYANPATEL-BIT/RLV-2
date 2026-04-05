# AGENT_CONTEXT.md
## READ THIS FIRST BEFORE DOING ANYTHING

This is a 24-hour hackathon project. Every session with an AI must start
by reading this file completely before writing any code or making suggestions.

This file is the RECOVERY DOCUMENT. If anything breaks the night before
submission, this file has every value, every design decision, and every
command needed to rebuild from scratch.

---

## What this project is

An OpenEnv environment for a DevOps/FinOps real-world task.
AI agents learn to optimize cloud infrastructure cost vs performance.
Deployed to HuggingFace Spaces. Must pass openenv validate.

Hackathon organizers: Meta + HuggingFace
Judging: Real-world utility 30%, Task quality 25%, Env design 20%,
         Code quality 15%, Creativity 10%

---

## Mandatory rules — never break these

- Stack: Python 3.11, FastAPI, Pydantic v2, uvicorn, OpenAI client ONLY
- Port: 7860 (HuggingFace default)
- inference.py MUST be in root directory, MUST be named exactly inference.py
- LLM calls MUST use OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars
- inference.py must complete in under 20 minutes on 2 vCPU / 8GB RAM
- No heavy ML dependencies in the environment itself
- No placeholders, no TODOs in any file — everything must be production-ready
- Graders must NEVER return the same score regardless of input
- Graders must ALWAYS return float 0.0–1.0, fully deterministic
- Do NOT create any new script files. The scripts/ folder contains exactly
  2 files owned by the course repo:
    - validate_notebooks.py  (course CI tool, do not touch)
    - validate_snippets.py   (course CI tool, do not touch)
  Never add anything to scripts/. Never create validation scripts anywhere
  in the project.
- Do NOT create separate models.py, tasks.py, utils.py, helpers.py, or tests/
  folders. Everything lives in env.py.

---

## File structure (FINAL — exactly these files)

project-root/
├── env.py                  # ALL-IN-ONE: models + tasks + graders + FastAPI app
├── inference.py            # baseline LLM agent (mandatory name + location)
├── openenv.yaml            # environment manifest (spec_version: 1)
├── Dockerfile              # python:3.11-slim, non-root user, port 7860
├── requirements-docker.txt # 5 packages: fastapi, uvicorn, pydantic, openai, requests
├── README.md               # full docs (all 10 required sections)
└── AGENT_CONTEXT.md        # this file (recovery document)

NOTE: models.py and tasks.py exist in the repo from earlier steps but are
ORPHANED — all logic is consolidated into env.py. They can be deleted.

---

## Environment domain

DevOps / FinOps — cloud infrastructure cost vs performance optimization.
The agent manages a simulated cloud environment and must make decisions
about resource allocation, scaling, and configuration to minimize cost
while maintaining performance SLAs.

---

## Pydantic models (all defined inside env.py)

### Instance catalog (6 tiers, power-of-2 scaling)
  nano:   1 CPU,  1 GB RAM, $0.05/hr
  micro:  1 CPU,  2 GB RAM, $0.10/hr
  small:  2 CPU,  4 GB RAM, $0.20/hr
  medium: 4 CPU,  8 GB RAM, $0.40/hr
  large:  8 CPU, 16 GB RAM, $0.80/hr
  xlarge: 16 CPU, 32 GB RAM, $1.60/hr

### ServerInfo fields
  server_id (str), instance_type (str), cpu_cores (int), ram_gb (int),
  cpu_utilization (float 0-1), ram_utilization (float 0-1),
  cost_per_hour (float), assigned_workloads (List[str])

### WorkloadInfo fields
  workload_id (str), name (str), required_cpu (float), required_ram (float),
  current_latency_ms (float), sla_latency_ms (float), is_critical (bool),
  assigned_server (Optional[str])

### Observation fields
  servers (List[ServerInfo]), workloads (List[WorkloadInfo]),
  total_cost_per_hour (float), budget_per_hour (float),
  sla_violations (int), unassigned_workloads (int),
  step_number (int), max_steps (int), done (bool), message (str)

### Action fields
  action_type: Literal["provision", "terminate", "resize", "migrate", "noop"]
  server_id: Optional[str]       — for terminate/resize
  instance_type: Optional[str]   — for provision/resize
  workload_id: Optional[str]     — for migrate
  target_server_id: Optional[str] — for migrate

### Reward fields
  score (float 0-1), cost_efficiency (float 0-1), performance_score (float 0-1),
  penalty (float 0-1), is_final (bool), breakdown (str)

### Reward formula
  score = (cost_efficiency × 0.50) + (performance_score × 0.40) − (penalty × 0.10)
  Clamped to [0.0, 1.0]

### Cost efficiency formula
  At/under budget: 0.80 + 0.20 × (1 − cost/budget)   → range 0.80–1.00
  Over budget:     0.80 − 0.40 × (cost−budget)/budget → degrades to 0.00 at 3× budget

### Performance score formula
  Weighted fraction of workloads assigned AND meeting SLA.
  Critical workloads carry 2× weight.

### Penalty sources
  Repeated identical action:              +0.10 per repeat
  Destructive termination (server w/ wl): +0.30 per occurrence
  3+ trailing noops:                      +0.05 per noop beyond 2
  2+ provisions without any migration:    +0.10

---

## Latency simulation model (inside env.py)

Latency is computed deterministically from resource pressure:

  total_cpu_demand = sum of required_cpu for all workloads on server
  total_ram_demand = sum of required_ram for all workloads on server
  base_util = max(cpu_demand/cpu_cores, ram_demand/ram_gb)
  overhead = 0.05 × (num_workloads_on_server − 1)   ← context-switching
  utilization = base_util + overhead

  utilization ≤ 0.5:  latency = SLA × 0.20            (comfortable)
  utilization 0.5–1.0: latency = SLA × (0.20 + t×0.55) (scales up)
  utilization > 1.0:   latency = SLA × (1.1 + excess×2.0) (BREACH)

Key consequence: a server at exactly 100% CPU with 1 workload → OK (75% of SLA).
But 100% CPU with 2 workloads → 105% utilization → SLA breach (context-switch overhead).
This is why Task 3's srv-001 (4 CPU, 2 workloads needing 4 CPU) breaches.

---

## 3 Tasks — exact starting states

### Task 1 — Easy: "Single Server Rightsizing"
  Budget: $0.20/hr | Max steps: 5

  Servers:
    srv-001: xlarge (16 CPU, 32 GB, $1.60/hr), workloads: [wl-web]

  Workloads:
    wl-web: "Company Website", 1.0 cpu, 2.0 GB, SLA 200ms, non-critical, on srv-001

  Initial: cost=$1.60, sla_violations=0, latency=40ms
  Optimal action: resize srv-001 to micro → cost=$0.10, latency=150ms (under SLA)
  Also good: resize to small → cost=$0.20, latency=40ms (at budget)

### Task 2 — Medium: "Multi-Service Consolidation"
  Budget: $0.50/hr | Max steps: 10

  Servers:
    srv-001: medium (4 CPU, 8 GB, $0.40/hr),  workloads: [wl-api]
    srv-002: large  (8 CPU, 16 GB, $0.80/hr),  workloads: [wl-db]
    srv-003: medium (4 CPU, 8 GB, $0.40/hr),  workloads: [wl-cache]
    srv-004: large  (8 CPU, 16 GB, $0.80/hr),  workloads: [wl-worker]

  Workloads:
    wl-api:    "API Gateway",       1.0 cpu, 2.0 GB, SLA 150ms, CRITICAL, on srv-001
    wl-db:     "Database Replica",  2.0 cpu, 4.0 GB, SLA 100ms, CRITICAL, on srv-002
    wl-cache:  "Redis Cache",       1.0 cpu, 1.5 GB, SLA 50ms,  non-crit, on srv-003
    wl-worker: "Background Worker", 0.5 cpu, 1.0 GB, SLA 500ms, non-crit, on srv-004

  Initial: cost=$2.40, sla_violations=0
  Total resource needs: 4.5 CPU, 8.5 GB RAM
  Best realistic: consolidate to 1 large ($0.80) — all SLAs met but over budget

### Task 3 — Hard: "Fleet Chaos Triage"
  Budget: $0.80/hr | Max steps: 15

  Servers:
    srv-001: medium (4 CPU, 8 GB, $0.40/hr),  workloads: [wl-auth, wl-api]  ← OVERLOADED
    srv-002: large  (8 CPU, 16 GB, $0.80/hr),  workloads: [wl-db]           ← oversized
    srv-003: nano   (1 CPU, 1 GB, $0.05/hr),  workloads: [wl-search]       ← UNDERSIZED
    srv-004: medium (4 CPU, 8 GB, $0.40/hr),  workloads: []                 ← IDLE WASTE
    srv-005: nano   (1 CPU, 1 GB, $0.05/hr),  workloads: [wl-metrics]      ← OK

  Workloads:
    wl-auth:    "Auth Service",         2.0 cpu, 3.0 GB, SLA 150ms,  CRITICAL, on srv-001 → BREACH (180ms)
    wl-api:     "API Gateway",          2.0 cpu, 2.0 GB, SLA 150ms,  CRITICAL, on srv-001 → BREACH (180ms)
    wl-db:      "Primary Database",     2.0 cpu, 4.0 GB, SLA 100ms,  CRITICAL, on srv-002 → OK (20ms)
    wl-search:  "Search Index",         1.0 cpu, 2.0 GB, SLA 100ms,  non-crit, on srv-003 → BREACH (310ms)
    wl-metrics: "Metrics Collector",    0.5 cpu, 0.5 GB, SLA 500ms,  non-crit, on srv-005 → OK (100ms)
    wl-ml-jobs: "ML Training Pipeline", 3.0 cpu, 4.0 GB, SLA 1000ms, non-crit, UNASSIGNED

  Initial: cost=$1.70, sla_violations=3, unassigned_workloads=1
  Total resource needs: 10.5 CPU, 15.5 GB RAM
  NOTE: srv-003 is NANO (not small) — wl-search needs 2 GB but nano only has 1 GB

---

## API endpoints

POST /reset  → body: {"task_id": "easy|medium|hard"}, returns Observation JSON
POST /step   → body: Action JSON, returns {observation, reward, done, info}
GET  /state  → no body, returns {task_id, observation, action_history, destructive_terminations}

Error handling:
  - Invalid action → step still counts, error message in observation.message
  - Step after done → returns done=True with error in info
  - Unknown task_id on reset → HTTP 400
  - No episode running on step → HTTP 400

---

## Environment variables required at runtime

For the environment server (env.py):
  (none — env.py has no external dependencies)

For inference (inference.py):
  API_BASE_URL  — LLM endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN      — HuggingFace API key (used as api_key for OpenAI client)
  ENV_URL       — environment server URL (default: http://localhost:7860)

---

## Expected baseline scores

| Task   | Difficulty | Baseline (LLM) | Random Agent | Optimal (theoretical) |
|--------|-----------|-----------------|--------------|----------------------|
| easy   | 🟢 Easy   | 0.80–0.85       | 0.30–0.40    | 0.90 (resize→micro)  |
| medium | 🟡 Medium | 0.40–0.55       | 0.15–0.25    | 0.68 (1 large)       |
| hard   | 🔴 Hard   | 0.15–0.30       | 0.05–0.15    | ~0.50 (full triage)  |

Validated grader scores (from smoke testing):
  Easy:   no-action=0.40, resize→micro=0.85, resize→small=0.80
  Medium: no-action=0.40, over-consolidate=0.55, 1-large=0.68
  Hard:   no-action=0.27, partial-fix=0.49, 5-noops=0.22

---

## Pre-submission checklist

[x] env.py contains all models, tasks, graders, and FastAPI app
[x] env.py starts on port 7860 with: python env.py
[x] POST /reset returns HTTP 200 with valid Observation
[x] POST /step returns {observation, reward, done, info}
[x] GET /state returns full state with action history
[x] All 3 graders return float 0.0–1.0, fully deterministic
[x] Graders return DIFFERENT scores for different inputs (verified 9 scenarios)
[x] inference.py uses OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN
[x] inference.py runs all 3 tasks sequentially and prints scores
[x] inference.py handles LLM errors (3 retries) and parse errors (fallback to noop)
[x] openenv.yaml has spec_version, name, tasks, observation_space, action_space
[x] Dockerfile uses python:3.11-slim, non-root user, port 7860
[x] requirements-docker.txt has only 5 packages (fastapi, uvicorn, pydantic, openai, requests)
[x] README.md has all 10 required sections
[x] No hardcoded API keys anywhere in code
[x] No placeholders or TODOs in any file

---

## Validation commands (copy-paste ready)

```bash
# 1. Start server locally
python env.py

# 2. Test reset endpoint (separate terminal)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# 3. Test step endpoint
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "resize", "server_id": "srv-001", "instance_type": "micro"}'

# 4. Test state endpoint
curl http://localhost:7860/state

# 5. Docker build + run
docker build -t openenv-devops .
docker run -p 7860:7860 openenv-devops

# 6. Run baseline inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token"
export ENV_URL="http://localhost:7860"
python inference.py

# 7. Validate openenv spec
pip install openenv-core
openenv validate

# 8. Verify HF Space is live (after deployment)
curl -s -o /dev/null -w "%{http_code}" -X POST $HF_SPACE_URL/reset
```

---

## Current status

[x] Step 1 — Pydantic models: DONE (consolidated into env.py)
[x] Step 2 — 3 tasks + graders: DONE (consolidated into env.py)
[x] Step 3 — env.py + FastAPI: DONE (all-in-one, smoke tested)
[x] Step 4 — openenv.yaml: DONE
[x] Step 5 — inference.py: DONE
[x] Step 6 — Dockerfile: DONE (+ requirements-docker.txt)
[x] Step 7 — README.md: DONE (all 10 required sections)
[x] Step 8 — AGENT_CONTEXT.md: DONE (this file, fully populated)

---

## Known issues / gotchas

1. ORPHANED FILES: models.py and tasks.py still exist in the repo from
   Steps 1-2. All their logic was consolidated into env.py in Step 3.
   They should be deleted before submission but won't break anything
   if left in place (nothing imports from them).

2. TASK 3 SRV-003: Changed from small ($0.20) to nano ($0.05) during
   Step 3 to make the wl-search SLA breach physically consistent with
   the latency simulation model. AGENT_CONTEXT originally said "small"
   but nano is what's actually in the code. Total cost is $1.70 not $1.85.

3. LATENCY AT EXACT CAPACITY: A single workload using exactly 100% of
   a server's resources (e.g. micro running a 1-CPU/2-GB workload) does
   NOT breach SLA — it runs at 75% of SLA. But TWO workloads at 100%
   total WILL breach due to 5% context-switching overhead. This is
   intentional and makes Task 3's srv-001 breach correctly.

4. COST EFFICIENCY SCORING: Being under budget does NOT give a flat 1.0.
   The formula gives 0.80 at budget and scales up to 1.0 as cost
   approaches zero. This ensures micro ($0.10) scores higher than
   small ($0.20) on Task 1. This was a deliberate fix from the initial
   design where both scored identically.

5. BUDGET IMPOSSIBILITY (TASK 3): Total resource needs (10.5 CPU, 15.5 GB)
   cannot fit on any single server under $0.80. Even an xlarge ($1.60)
   is over budget. This is intentional — the agent gets partial credit
   for cost reduction even if it can't reach the budget. The theoretical
   optimal score is ~0.50, not 1.0.

6. INFERENCE CONTEXT WINDOW: inference.py trims conversation history
   to the last 6 exchanges (12 messages + system prompt) to avoid
   token overflow on smaller models. This means the LLM "forgets"
   early actions on Task 3 (15 steps).

7. NO EARLY TERMINATION: Episodes always run to max_steps. The agent
   should use noop if it believes it's done. noop is only penalized
   if used 3+ times in a row at the end.

---

## Emergency recovery steps

If env.py gets corrupted, the key values to reconstruct are:

1. Instance catalog: {nano: 1/1/0.05, micro: 1/2/0.10, small: 2/4/0.20,
   medium: 4/8/0.40, large: 8/16/0.80, xlarge: 16/32/1.60}
2. Reward: 0.50×cost_eff + 0.40×perf − 0.10×penalty, clamped [0,1]
3. Cost efficiency: under budget → 0.80 + 0.20×savings_ratio;
   over budget → 0.80 − 0.40×overspend_ratio
4. Performance: weighted fraction (critical=2×) of assigned+SLA-met workloads
5. Latency: SLA × f(utilization), breach at util > 1.0, +5% overhead per extra workload
6. Task configs: easy=5 steps/$0.20, medium=10/$0.50, hard=15/$0.80

The complete env.py is ~500 lines. Start with the models, then tasks,
then grading functions, then CloudEnvironment class, then FastAPI endpoints.