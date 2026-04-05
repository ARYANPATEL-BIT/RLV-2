# AGENT_CONTEXT.md
## READ THIS FIRST BEFORE DOING ANYTHING

This is a 24-hour hackathon project. Every session with an AI must start
by reading this file completely before writing any code or making suggestions.

---

## What this project is

An OpenEnv environment for a DevOps/FinOps real-world task (v2.0).
AI agents learn to optimize cloud infrastructure cost vs performance
with 4 resource dimensions, spot eviction, cascading failures, and
procedural task generation. Deployed to HuggingFace Spaces.

Hackathon organizers: Meta + HuggingFace
Judging: Real-world utility 30%, Task quality 25%, Env design 20%,
         Code quality 15%, Creativity 10%

---

## Mandatory rules — never break these

- Stack: Python 3.11, FastAPI, Pydantic v2, uvicorn, OpenAI client ONLY
- Port: 7860 (HuggingFace default)
- inference.py MUST be in root directory, MUST be named exactly inference.py
- LLM calls MUST use OpenAI client with API_BASE_URL, MODEL_NAME, HF_TOKEN
- inference.py must complete in under 20 minutes on 2 vCPU / 8GB RAM
- No heavy ML dependencies in the environment itself
- No placeholders, no TODOs in any file
- Graders must return float 0.0–1.0, fully deterministic
- Graders must return DIFFERENT scores for different inputs

---

## File structure (FINAL — exactly these files)

project-root/
├── env.py                  # ALL-IN-ONE: models + tasks + graders + dashboard + FastAPI
├── inference.py            # baseline LLM agent (mandatory name + location)
├── test_env.py             # 14 test suites with 50+ assertions
├── openenv.yaml            # environment manifest (spec_version: 1)
├── Dockerfile              # python:3.11-slim, non-root user, port 7860
├── requirements.txt        # 5 project-specific packages
├── requirements-docker.txt # 5 packages for Docker build
├── README.md               # full docs (all 10 required sections)
└── AGENT_CONTEXT.md        # this file (recovery document)

---

## Key design decisions (v2.0)

### Instance catalog (11 types)
  nano:        1cpu,  1GB,  1k iops, 0.5Gbps, $0.05
  micro:       1cpu,  2GB,  2k iops, 0.5Gbps, $0.10
  small:       2cpu,  4GB,  3k iops, 1.0Gbps, $0.20
  medium:      4cpu,  8GB,  5k iops, 2.0Gbps, $0.40
  large:       8cpu, 16GB, 10k iops, 5.0Gbps, $0.80
  xlarge:     16cpu, 32GB, 20k iops, 10Gbps,  $1.60
  compute-opt: 8cpu,  8GB,  5k iops, 5.0Gbps, $0.60
  memory-opt:  4cpu, 32GB, 10k iops, 2.0Gbps, $0.70
  storage-opt: 4cpu,  8GB, 30k iops, 2.0Gbps, $0.55
  spot-medium: 4cpu,  8GB,  5k iops, 2.0Gbps, $0.12 (spot!)
  spot-large:  8cpu, 16GB, 10k iops, 5.0Gbps, $0.24 (spot!)

### Reward formula
  score = ce×0.40 + perf×0.35 - pen×0.25
  Clamped [0, 1]. If perf=0, ce is zeroed (documented).
  Cost efficiency: 1.0 at zero cost, 0.90 at budget, 0.0 at 2x budget.

### Penalty amounts
  Repeated non-noop action: +0.15 per repeat
  Destructive termination: +0.50 per occurrence (severe!)
  All-noop idleness: +0.05 per noop beyond 2 (only if agent did NOTHING useful)
  2+ provisions no migration: +0.10
  Consecutive non-noop loop (3+): +0.10 per iteration

### Novel mechanics
  1. Spot eviction: at steps 3, 7, 12. Hash-based, 40% per spot server.
  2. Cascading failures: dependency list on workloads. 1.5x latency per failing dep.
  3. Traffic spikes: seed-based multiplier [0.85, 1.45] on resource demands.
  4. 4D latency: bottleneck = max(cpu%, ram%, disk%, net%) + context-switch overhead.

### Session management
  /reset returns session_id. /step accepts ?session_id=. Falls back to latest session.

### Dashboard
  GET /dashboard?session_id=ID returns live HTML with server cards, utilization bars, workload status.

---

## 3 Tasks — exact starting states

### Task 1 — Easy: "Single Server Rightsizing"
  Budget: $0.25/hr | Max steps: 5
  Servers: srv-001: xlarge ($1.60), workloads: [wl-web, wl-api]
  Workloads:
    wl-web: 0.5cpu, 1.0GB, 500iops, 0.1Gbps, SLA 200ms, non-critical
    wl-api: 1.0cpu, 1.5GB, 800iops, 0.3Gbps, SLA 150ms, CRITICAL
  Optimal: resize to small ($0.20). micro would breach (1.5cpu > 1cpu).

### Task 2 — Medium: "Multi-Service Consolidation"
  Budget: $0.50/hr | Max steps: 10
  4 servers (2 medium, 2 large) = $2.40/hr
  wl-api depends on wl-db (cascading failure)
  Total needs: 4.5 CPU, 8.5 GB RAM

### Task 3 — Hard: "Fleet Chaos Triage"
  Budget: $1.20/hr | Max steps: 15
  5 servers, 6 workloads, $1.70/hr starting cost
  SLA violations: 3 (wl-auth, wl-api, wl-search)
  Unassigned: 1 (wl-ml-jobs)
  Dependencies: wl-api→wl-auth, wl-search→wl-db
  Optimal: large + medium = $1.20/hr (achievable!)

---

## Pre-submission checklist

[x] env.py contains all models, tasks, graders, dashboard, and FastAPI app
[x] env.py starts on port 7860 with: python env.py
[x] POST /reset returns session_id + Observation
[x] POST /step returns {observation, reward, done, info}
[x] GET /state returns full state
[x] GET /dashboard returns live HTML dashboard
[x] All graders return float 0.0–1.0, fully deterministic
[x] Graders return DIFFERENT scores for different inputs (verified in tests)
[x] 11 instance types including spot and specialized
[x] 4 resource dimensions (CPU, RAM, disk IOPS, network)
[x] Spot eviction at steps 3/7/12 (deterministic)
[x] Cascading failures via workload dependencies
[x] Traffic spikes via seed-based multiplier
[x] Procedural task generation via task_id="random" + seed
[x] Session management with auto-cleanup
[x] inference.py handles sessions, new instance types, new mechanics
[x] test_env.py has 14 test suites with 50+ assertions
[x] openenv.yaml has all new fields documented
[x] Dockerfile uses python:3.11-slim, non-root user, port 7860
[x] requirements.txt has only project-specific packages
[x] README.md has all 10 sections + novel mechanics section
[x] No orphaned files (models.py, tasks.py deleted)
[x] No hardcoded API keys
[x] No placeholders or TODOs

---

## Current status

[x] All 23 review fixes implemented
[x] 14 test suites passing (50+ assertions)
[x] Ready for submission