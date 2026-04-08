---
title: FinOps Cloud Optimizer Environment
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - devops
  - finops
  - reinforcement-learning
  - agents
---

# ☁️ DevOps/FinOps Cloud Optimizer — OpenEnv Environment

**An AI agent environment where models learn to optimize cloud infrastructure cost vs. performance under realistic operational constraints.**

---

## 1. Environment Description & Motivation

Cloud infrastructure is the single largest variable cost for most software companies. In 2025, global cloud spend exceeded $800B — and an estimated 30% of it is wasted on over-provisioned, idle, or misconfigured resources (Flexera State of the Cloud Report).

**FinOps** (Financial Operations) is the discipline of making cloud spending decisions based on data. This environment simulates a fleet of cloud servers running production workloads across **4 resource dimensions** (CPU, RAM, disk IOPS, network bandwidth). The agent observes real-time utilization, latency, cost, and SLA status — then takes infrastructure actions to minimize cost while keeping everything healthy.

### What makes this environment unique

- **4-dimensional resource model:** CPU, RAM, disk IOPS, and network bandwidth. A server can be overloaded on any dimension, creating non-obvious bottlenecks.
- **11 instance types** including specialized tiers (compute-optimized, memory-optimized, storage-optimized) and **spot instances** that are 70% cheaper but can be deterministically evicted.
- **Cascading failures:** Workloads can depend on other workloads. If a dependency breaches SLA or goes down, the dependent's latency inflates by 1.5×.
- **Deterministic traffic spikes:** When a seed is provided, workload demands fluctuate per step (range 0.85×–1.45×), testing adaptive reasoning.
- **Procedural task generation:** Beyond the 3 fixed tasks, unlimited evaluation scenarios via `task_id="random"` + seed.
- **Session management:** Multiple concurrent agents can run independently via session IDs.

### Why this matters for RL/GRPO training

- **Multi-objective optimization:** Cost and performance are in tension. The reward function gives continuous signal with partial credit across the full trajectory.
- **Sequential decision-making with ordering constraints:** You can't terminate before migrating. You can't migrate without free capacity. Actions must be sequenced correctly.
- **Novel mechanics:** Spot eviction forces planning around failure modes. Cascading failures require understanding dependency graphs. Traffic spikes require adaptive resource allocation.
- **Generalization pressure:** Procedural generation prevents memorization. Agents must learn transferable strategies.
- **Deterministic grading:** Same actions always produce the same score. No randomness, no hidden state.

---

## 2. Observation Space

Every step, the agent receives a full snapshot of the infrastructure:

| Field | Type | Description |
|---|---|---|
| `servers` | `List[ServerInfo]` | All active server instances (see sub-fields) |
| `workloads` | `List[WorkloadInfo]` | All workloads that must be kept running |
| `total_cost_per_hour` | `float` | Sum of all running server costs in USD/hr |
| `budget_per_hour` | `float` | Target budget the agent should optimize toward |
| `sla_violations` | `int` | Workloads currently breaching latency SLA |
| `unassigned_workloads` | `int` | Workloads not running on any server (DOWN) |
| `step_number` | `int` | Current step in episode (0-indexed) |
| `max_steps` | `int` | Maximum steps before episode ends |
| `done` | `bool` | True if the episode is over |
| `message` | `str` | Structured feedback: `[OK]`, `[ERROR:*]`, `[WARN:*]`, `[DESTRUCTIVE]`, `[EVICTION]` |
| `traffic_multiplier_active` | `bool` | True if seed-based traffic fluctuation is active |
| `spot_eviction_occurred` | `bool` | True if a spot instance was evicted this step |

### ServerInfo sub-fields

| Field | Type | Description |
|---|---|---|
| `server_id` | `str` | Unique identifier (e.g. `"srv-001"`) |
| `instance_type` | `str` | One of 11 instance types (see catalog below) |
| `cpu_cores` | `int` | Number of vCPU cores |
| `ram_gb` | `int` | RAM in GB |
| `disk_iops` | `int` | Disk I/O operations per second |
| `network_gbps` | `float` | Network bandwidth in Gbps |
| `cpu_utilization` | `float` | Current CPU usage (**uncapped** — can exceed 1.0 when overloaded) |
| `ram_utilization` | `float` | Current RAM usage (uncapped) |
| `disk_utilization` | `float` | Current disk usage (uncapped) |
| `network_utilization` | `float` | Current network usage (uncapped) |
| `cost_per_hour` | `float` | Hourly cost in USD |
| `assigned_workloads` | `List[str]` | Workload IDs running on this server |
| `is_spot` | `bool` | True if spot instance (can be evicted) |
| `is_overloaded` | `bool` | True if any resource dimension exceeds 100% |

### WorkloadInfo sub-fields

| Field | Type | Description |
|---|---|---|
| `workload_id` | `str` | Unique identifier (e.g. `"wl-api"`) |
| `name` | `str` | Human-readable name |
| `required_cpu` | `float` | Minimum CPU cores needed |
| `required_ram` | `float` | Minimum RAM in GB |
| `required_disk_iops` | `int` | Minimum disk IOPS needed |
| `required_net_gbps` | `float` | Minimum network Gbps needed |
| `current_latency_ms` | `float` | Current p95 latency in ms |
| `sla_latency_ms` | `float` | Maximum acceptable latency |
| `is_critical` | `bool` | If true, SLA breach carries 2× penalty weight |
| `assigned_server` | `str \| null` | Server ID, or `null` if unassigned |
| `dependencies` | `List[str]` | Workload IDs this workload depends on |

### Instance Type Catalog (11 types)

| Type | CPU | RAM | Disk IOPS | Network | Cost/hr | Notes |
|---|---|---|---|---|---|---|
| `nano` | 1 | 1 GB | 1,000 | 0.5 Gbps | $0.05 | Minimal |
| `micro` | 1 | 2 GB | 2,000 | 0.5 Gbps | $0.10 | Light workloads |
| `small` | 2 | 4 GB | 3,000 | 1.0 Gbps | $0.20 | General purpose |
| `medium` | 4 | 8 GB | 5,000 | 2.0 Gbps | $0.40 | General purpose |
| `large` | 8 | 16 GB | 10,000 | 5.0 Gbps | $0.80 | Heavy workloads |
| `xlarge` | 16 | 32 GB | 20,000 | 10.0 Gbps | $1.60 | Maximum capacity |
| `compute-opt` | 8 | 8 GB | 5,000 | 5.0 Gbps | $0.60 | High CPU, low RAM |
| `memory-opt` | 4 | 32 GB | 10,000 | 2.0 Gbps | $0.70 | High RAM |
| `storage-opt` | 4 | 8 GB | 30,000 | 2.0 Gbps | $0.55 | High disk IOPS |
| `spot-medium` | 4 | 8 GB | 5,000 | 2.0 Gbps | **$0.12** | ⚡ 70% cheaper, **can be evicted** |
| `spot-large` | 8 | 16 GB | 10,000 | 5.0 Gbps | **$0.24** | ⚡ 70% cheaper, **can be evicted** |

---

## 3. Action Space

Each step, the agent submits exactly one action:

| Action | Required Fields | Effect | Risk |
|---|---|---|---|
| `provision` | `instance_type` | Creates a new empty server | Increases cost without benefit if nothing migrated to it |
| `terminate` | `server_id` | Destroys a server permanently | **Orphans all workloads** — 0.50 penalty per occurrence |
| `resize` | `server_id`, `instance_type` | Changes instance type in-place | May cause SLA breach if downsized too aggressively |
| `migrate` | `workload_id`, `target_server_id` | Moves a workload to another server | Target may become overloaded. Capacity warnings shown |
| `noop` | — | Does nothing | Wastes a step. 3+ consecutive noops incur escalating penalty |

---

## 4. Reward Function

```
score = (cost_efficiency × 0.40) + (performance × 0.35) − (penalty × 0.25)
```

Clamped to **[0.0, 1.0]**. Fully deterministic. If performance = 0 (all workloads unhealthy), cost_efficiency is zeroed — you can't score well by just cutting costs.

### Cost Efficiency (40% weight)

| Condition | Score |
|---|---|
| Zero cost | 1.00 |
| 50% of budget | 0.95 |
| At budget | 0.90 |
| 1.5× budget | 0.45 |
| 2× budget or more | 0.00 |

### Performance Score (35% weight)

Weighted fraction of workloads that are assigned AND meeting SLA. Critical workloads carry 2× weight.

### Penalty (25% weight — severe)

| Source | Amount | Notes |
|---|---|---|
| Repeated identical non-noop action | +0.15 per repeat | Prevents loops (noops excluded) |
| Consecutive non-noop loop (3+) | +0.10 per iteration | Extra penalty for action loops |
| **Destructive termination** | **+0.50** per occurrence | Orphaning workloads is catastrophic |
| All-noop idleness | +0.05 per noop beyond 2 | Only if agent did zero useful actions |
| 2+ provisions without migration | +0.10 | Prevents mindless provisioning |

---

## 5. Novel Mechanics

### Spot Instance Eviction

Spot instances (`spot-medium`, `spot-large`) cost 70% less but can be **evicted** at steps 3, 7, and 12. Eviction is deterministic (based on server ID hash), so agents can learn to predict and plan around it. Evicted servers are removed; all workloads on them become orphaned.

### Cascading Failures

Workloads can declare `dependencies`. If a dependency is DOWN (unassigned) or breaching SLA, the dependent workload's latency is inflated by 1.5× per failing dependency. This creates chains where fixing one root cause resolves multiple SLA violations.

### Traffic Spikes

When a `seed` is provided at `/reset`, workload resource demands fluctuate per step (deterministic range: 0.85×–1.45×). This forces agents to build in headroom rather than right-sizing to exact capacity.

### 4D Resource Bottlenecks

Latency is driven by the **worst** resource dimension: `max(CPU%, RAM%, disk%, network%)`. A server with low CPU but saturated disk IOPS will breach SLA just like a CPU-overloaded one.

---

## 6. Tasks

### Task 1: Single Server Rightsizing (Easy)

| | |
|---|---|
| **Max Steps** | 5 |
| **Budget** | $0.25/hr |
| **Objective** | Resize an xlarge server ($1.60/hr) running 2 workloads (0.5+1.0 CPU, 1.0+1.5 GB RAM) to the smallest instance that keeps both SLAs met |

**What a good agent does:** Resizes to `small` (2 CPU, 4 GB). Both workloads fit. Cost drops from $1.60 to $0.20.

**Why it's not trivial:** Agent must verify BOTH workloads fit — `micro` (1 CPU) causes CPU overload and SLA breach.

### Task 2: Multi-Service Consolidation (Medium)

| | |
|---|---|
| **Max Steps** | 10 |
| **Budget** | $0.50/hr |
| **Objective** | Consolidate 4 workloads from 4 oversized servers ($2.40/hr) with cascading dependency (wl-api→wl-db) |

### Task 3: Fleet Chaos Triage (Hard)

| | |
|---|---|
| **Max Steps** | 15 |
| **Budget** | $1.20/hr |
| **Objective** | Fix a broken fleet — 3 SLA breaches, 1 orphaned workload, cascading failures (wl-api→wl-auth, wl-search→wl-db), 1 idle server |

**Optimal:** Consolidate onto `large` + `medium` ($1.20/hr). Requires 8–12 carefully ordered actions.

### Task 4: Procedural Generation (Variable)

```bash
curl -X POST http://localhost:7860/reset \
  -d '{"task_id": "random", "seed": 42}'
```

Generates a unique fleet configuration from the seed. Server count (3–7), workload count (4–8), resource requirements, dependencies, and budget are all randomized. Same seed always produces the same scenario.

---

## 7. Local Setup

```bash
# 1. Clone and set up
git clone https://github.com/ARYANPATEL-BIT/RLV-2.git && cd RLV-2
python -m venv venv && source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Run tests
python test_env.py

# 3. Start environment
python env.py
# Server starts at http://localhost:7860

# 4. Test (separate terminal)
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "easy"}'

# 5. View dashboard
# Open http://localhost:7860/dashboard?session_id=YOUR_SESSION_ID
```

---

## 8. Docker Setup

```bash
docker build -t openenv-devops .
docker run -p 7860:7860 openenv-devops
curl -X POST http://localhost:7860/reset -d '{"task_id": "easy"}'
```

---

## 9. Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"
python inference.py
```

Typical runtime: **2–3 minutes** on 2 vCPU / 8 GB RAM.

---

## 10. Baseline Scores

| Task | Difficulty | Max Steps | Budget | Baseline (LLM) | Random Agent |
|---|---|---|---|---|---|
| Single Server Rightsizing | 🟢 Easy | 5 | $0.25/hr | 0.65–0.75 | ~0.15–0.30 |
| Multi-Service Consolidation | 🟡 Medium | 10 | $0.50/hr | 0.40–0.55 | ~0.10–0.20 |
| Fleet Chaos Triage | 🔴 Hard | 15 | $1.20/hr | 0.25–0.40 | ~0.05–0.15 |

All scores are deterministic: same actions → same score, always.

---

## API Reference

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/reset` | POST | `{"task_id": "easy\|medium\|hard\|random", "seed": <int>}` | `{session_id, ...Observation}` |
| `/step` | POST | `Action JSON` + `?session_id=ID` | `{observation, reward, done, info}` |
| `/state` | GET | `?session_id=ID` | Full state with action history |
| `/info` | GET | — | Task catalog, instance types, reward formula |
| `/dashboard` | GET | `?session_id=ID` | Live HTML dashboard |

---

## Example Episode (Easy Task)

A complete episode walkthrough showing reset, one action, and the final state:

```bash
# 1. Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

**Response:** An xlarge server ($1.60/hr) running two workloads. Budget is $0.25/hr.

```json
{
  "session_id": "a1b2c3d4",
  "servers": [{"server_id": "srv-001", "instance_type": "xlarge", "cpu_cores": 16, "ram_gb": 32, "cost_per_hour": 1.6, "assigned_workloads": ["wl-web", "wl-api"]}],
  "workloads": [
    {"workload_id": "wl-web", "name": "Company Website", "required_cpu": 0.5, "required_ram": 1.0, "current_latency_ms": 40.0, "sla_latency_ms": 200.0, "is_critical": false},
    {"workload_id": "wl-api", "name": "API Gateway", "required_cpu": 1.0, "required_ram": 1.5, "current_latency_ms": 30.0, "sla_latency_ms": 150.0, "is_critical": true}
  ],
  "total_cost_per_hour": 1.6, "budget_per_hour": 0.25, "sla_violations": 0,
  "step_number": 0, "max_steps": 5, "done": false
}
```

```bash
# 2. Resize the server down to small (2 CPU, 4 GB — fits both workloads)
curl -X POST "http://localhost:7860/step?session_id=a1b2c3d4" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "resize", "server_id": "srv-001", "instance_type": "small"}'
```

**Response:** Cost drops from $1.60 to $0.20. SLAs still met. Score 0.718.

```json
{
  "observation": {"total_cost_per_hour": 0.2, "sla_violations": 0, "step_number": 1},
  "reward": {"score": 0.718, "cost_efficiency": 0.92, "performance_score": 1.0, "penalty": 0.0,
             "breakdown": "cost_eff=0.920×0.4 + perf=1.000×0.35 - penalty=0.000×0.25 = 0.7180"},
  "done": false
}
```

---

## Technical Stack

- **Python 3.11** + **FastAPI** + **Pydantic v2** + **uvicorn**
- **OpenAI client** for LLM inference (any OpenAI-format API)
- **Zero ML dependencies** in the environment itself
- **Fully deterministic** grading — no randomness anywhere
- **Single-file environment** (`env.py`) — models, tasks, grading, server, and dashboard
- **14 test suites** (`test_env.py`) with 50+ assertions covering all mechanics
