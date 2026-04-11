---
title: OpenEnv DevOps FinOps Cloud Optimizer
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# ☁️ OpenEnv: DevOps/FinOps Cloud Optimizer

A production-grade **cloud infrastructure cost-optimization environment** built on the [OpenEnv specification](https://github.com/meta-pytorch/openenv). An RL agent observes live fleet metrics — CPU, RAM, disk I/O, network — and manages servers to minimize cost while meeting SLA latency targets. Includes task-based evaluation with deterministic graders (0.0–1.0 scoring).

Designed to evaluate both reinforcement learning policies and LLM-based decision agents.

This environment is fully compliant with OpenEnv and supports both local Docker execution and remote evaluation via Hugging Face Spaces.

---

## Real-World Motivation

Every company running workloads on AWS, GCP, or Azure faces the same challenge: **cloud bills grow faster than revenue**. Gartner estimates that organizations waste over 30% of their cloud spend on idle or over-provisioned resources. FinOps teams at Netflix, Spotify, and Uber manually triage dashboards to right-size instances — a process that is slow, error-prone, and never-ending:

- **Over-provisioned servers** waste money on idle CPU and RAM.
- **Under-provisioned servers** breach SLA latency targets, cascading failures to dependent services.
- **Spot instances** offer 70% savings but can be evicted without warning.
- **Manual optimization** cannot keep pace with dynamic traffic patterns.

This environment models the FinOps problem as an RL task: the agent receives real-time fleet telemetry and must learn a policy that generalises across infrastructure scenarios of increasing difficulty.

---

## RL Loop

```
┌───────────┐   observation   ┌─────────────┐   action   ┌───────────┐
│   Agent   │ ◄───────────── │  Cloud Env  │ ◄──────── │   Agent   │
│   (LLM)   │ ─────────────► │  Simulator  │ ────────► │  Decision │
└───────────┘   reward        └─────────────┘           └───────────┘
```

1. **Observe**: server utilization (CPU/RAM/disk/net), workload latency, cost, SLA status
2. **Act**: provision, terminate, resize, migrate, or noop
3. **Reward**: composite signal balancing cost efficiency, SLA performance, and penalties
4. **Repeat** for 5–15 steps per episode (task-dependent)

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `servers` | `List[ServerInfo]` | All active server instances with capacity, utilization, cost |
| `workloads` | `List[WorkloadInfo]` | All workloads with resource demands, latency, SLA targets |
| `total_cost_per_hour` | `float` | Sum of all running server costs (USD/hr) |
| `budget_per_hour` | `float` | Target budget the agent should optimize toward |
| `sla_violations` | `int` | Count of workloads breaching their latency SLA |
| `unassigned_workloads` | `int` | Count of workloads not running on any server (DOWN) |
| `step_number` | `int` | Current step in the episode |
| `max_steps` | `int` | Maximum steps before episode ends |
| `done` | `bool` | Whether the episode is over |
| `message` | `str` | Structured feedback: `[OK]`, `[ERROR:*]`, `[DESTRUCTIVE]`, `[EVICTION]` |
| `traffic_multiplier_active` | `bool` | Whether seed-based traffic fluctuation is active |
| `spot_eviction_occurred` | `bool` | Whether a spot instance was evicted this step |

---

## Action Space

| Action | Required Params | Description |
|---|---|---|
| `provision` | `instance_type` | Spawn a new server from the 11-type catalog |
| `terminate` | `server_id` | Destroy a server. **Danger:** orphans workloads (0.50 penalty) |
| `resize` | `server_id`, `instance_type` | Change instance tier in-place (workloads stay) |
| `migrate` | `workload_id`, `target_server_id` | Move a workload to another server |
| `noop` | — | Do nothing (wastes a step) |

### Instance Catalog (11 types)

| Type | CPU | RAM | Disk IOPS | Network | Cost/hr | Spot? |
|---|---|---|---|---|---|---|
| `nano` | 1 | 1 GB | 1,000 | 0.5 Gbps | $0.05 | No |
| `micro` | 1 | 2 GB | 2,000 | 0.5 Gbps | $0.10 | No |
| `small` | 2 | 4 GB | 3,000 | 1.0 Gbps | $0.20 | No |
| `medium` | 4 | 8 GB | 5,000 | 2.0 Gbps | $0.40 | No |
| `large` | 8 | 16 GB | 10,000 | 5.0 Gbps | $0.80 | No |
| `xlarge` | 16 | 32 GB | 20,000 | 10.0 Gbps | $1.60 | No |
| `compute-opt` | 8 | 8 GB | 5,000 | 5.0 Gbps | $0.60 | No |
| `memory-opt` | 4 | 32 GB | 10,000 | 2.0 Gbps | $0.70 | No |
| `storage-opt` | 4 | 8 GB | 30,000 | 2.0 Gbps | $0.55 | No |
| `spot-medium` | 4 | 8 GB | 5,000 | 2.0 Gbps | $0.12 | ⚡ Yes |
| `spot-large` | 8 | 16 GB | 10,000 | 5.0 Gbps | $0.24 | ⚡ Yes |

---

## Tasks

Three tasks of increasing difficulty, plus procedural generation for unlimited evaluation scenarios. Each task uses a deterministic grader returning a score in [0.0, 1.0].

### Task 1 — Easy: Single Server Rightsizing

| | |
|---|---|
| **Budget** | $0.25/hr |
| **Max Steps** | 5 |
| **Workload** | Static — 2 workloads on 1 over-provisioned xlarge ($1.60/hr) |
| **Goal** | Resize to the smallest instance fitting both workloads |
| **Grader** | `score = clamp(cost_reduction / max_possible_reduction, 0.001, 0.998)` |

### Task 2 — Medium: Multi-Service Consolidation

| | |
|---|---|
| **Budget** | $0.50/hr |
| **Max Steps** | 10 |
| **Workload** | 4 workloads on 4 oversized servers ($2.40/hr), with cascading dependency |
| **Goal** | Consolidate onto fewer right-sized servers while maintaining SLAs |
| **Grader** | `score = 0.5 × cost_score + 0.5 × sla_score` |

### Task 3 — Hard: Fleet Chaos Triage

| | |
|---|---|
| **Budget** | $1.20/hr |
| **Max Steps** | 15 |
| **Workload** | Broken fleet: 3 SLA breaches, 1 orphaned workload, cascading failures |
| **Goal** | Triage and fix everything within budget |
| **Grader** | `score = 0.4 × cost + 0.3 × sla + 0.2 × efficiency + 0.1 × stability` |

```
Hard grader (detailed):
score = 0.4 × cost_efficiency(total_cost, budget)
      + 0.3 × performance_score(workloads)
      + 0.2 × (1.0 − penalty_ratio)
      + 0.1 × stability_bonus
```

All graders enforce strict open-interval bounds (0, 1) — scores never reach exactly 0.0 or 1.0.

---

## Reward Function

Per-step reward (continuous, non-sparse):

```
score = cost_efficiency × 0.40
      + performance    × 0.35
      − penalty        × 0.25
```

Clamped to [0.0, 1.0]. If performance = 0 (all workloads unhealthy), cost efficiency is zeroed — you cannot score well by simply terminating everything.

**Penalty sources:**
- Destructive termination (workloads orphaned): **+0.50 per occurrence**
- Repeated non-noop actions: +0.15 per repeat
- All-noop idleness (3+ noops, no useful actions): +0.05 per extra noop
- Provision spam (2+ provisions, no migrations): +0.10
- Consecutive action loop (3+): +0.10 per iteration

---

## Baseline Results

| Task | Difficulty | Max Steps | Budget | Baseline Score |
|---|---|---|---|---|
| **Rightsizing** | 🟢 Easy | 5 | $0.25/hr | **0.847** |
| **Consolidation** | 🟡 Medium | 10 | $0.50/hr | **0.623** |
| **Chaos Triage** | 🔴 Hard | 15 | $1.20/hr | **0.412** |

Baseline uses an LLM agent (Llama 3.1 8B via HuggingFace Router) with deterministic grading. All scores are computed using the same reward formula: `cost×0.40 + perf×0.35 − penalty×0.25`.

---

## Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server (port 7860)
python env.py

# Server runs at http://localhost:7860
# Interactive playground at http://localhost:7860
# API docs at http://localhost:7860/docs
```

### Docker

```bash
# Build
docker build -t openenv-devops .

# Run
docker run -p 7860:7860 openenv-devops
```

### Python Client

```python
from client import DevOpsEnv

env = DevOpsEnv("https://aryan-void-openenv-devops-2.hf.space")
obs = env.reset(task_id="easy")

# Take an action
result = env.step({
    "action_type": "resize",
    "server_id": "srv-001",
    "instance_type": "small"
})
print(f"Score: {result['reward']['score']}")
```

### cURL

```bash
# Reset
curl -X POST https://aryan-void-openenv-devops-2.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'

# Step
curl -X POST "https://aryan-void-openenv-devops-2.hf.space/step?session_id=SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "resize", "server_id": "srv-001", "instance_type": "small"}'

# Get State
curl "https://aryan-void-openenv-devops-2.hf.space/state?session_id=SESSION_ID"

# Health Check
curl https://aryan-void-openenv-devops-2.hf.space/health
```

### Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_huggingface_token"

python inference.py
```

---

## Live Deployment

👉 **https://aryan-void-openenv-devops-2.hf.space**

### Health Check
```bash
curl https://aryan-void-openenv-devops-2.hf.space/health
# {"status":"ok","environment":"devops-finops-cloud-optimizer","version":"2.0.0"}
```

### Reset Example
```bash
curl -X POST https://aryan-void-openenv-devops-2.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

### Step Example
```bash
curl -X POST "https://aryan-void-openenv-devops-2.hf.space/step?session_id=YOUR_ID" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "noop"}'
```

---

## Project Structure

```
openenv-devops-2/
├── env.py                    # All-in-one: models, tasks, graders, env, API, playground
├── inference.py              # Baseline LLM agent (mandatory name + location)
├── models.py                 # Pydantic v2 models (re-exports from env.py)
├── tasks.py                  # Task definitions & grader functions
├── client.py                 # Python client wrapper (DevOpsEnv)
├── test_env.py               # 14 test suites, 78 assertions
├── openenv.yaml              # OpenEnv environment manifest
├── Dockerfile                # python:3.11-slim, non-root, port 7860
├── requirements.txt          # Project dependencies
├── requirements-docker.txt   # Docker build dependencies
├── README.md                 # This file
├── AGENT_CONTEXT.md          # Agent recovery context
└── server/
    ├── __init__.py
    └── app.py                # FastAPI server entry point
```

---

## Novel Environment Mechanics

- **4-Dimensional Resource Model**: Bottleneck = max(CPU%, RAM%, disk%, network%) + context-switch overhead
- **Spot Instance Eviction**: Deterministic at steps 3, 7, 12 — 40% chance per spot server
- **Cascading Failures**: Workload dependencies (e.g. API→DB) — if a dependency breaches SLA, dependent latency inflates 1.5×
- **Traffic Spikes**: Seed-based demand multiplier [0.85, 1.45] on per-workload resource requirements
- **Procedural Generation**: `task_id="random"` + `seed` for unlimited unique fleet configurations
