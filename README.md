# ☁️ DevOps/FinOps Cloud Optimizer — OpenEnv Environment

**An AI agent environment where models learn to optimize cloud infrastructure cost vs. performance under realistic operational constraints.**

---

## 1. Environment Description & Motivation

Cloud infrastructure is the single largest variable cost for most software companies. In 2025, global cloud spend exceeded $800B — and an estimated 30% of it is wasted on over-provisioned, idle, or misconfigured resources (Flexera State of the Cloud Report).

**FinOps** (Financial Operations) is the discipline of making cloud spending decisions based on data. Today, these decisions are made manually by engineers staring at dashboards. The gap between "what we're spending" and "what we should be spending" is exactly the kind of problem AI agents can solve — if they can learn to reason about:

- **Resource right-sizing:** Matching server capacity to actual workload demand
- **Consolidation:** Packing multiple workloads onto fewer servers efficiently
- **SLA management:** Maintaining latency guarantees while cutting costs
- **Triage under chaos:** Diagnosing and fixing broken infrastructure with incomplete information

This environment simulates a fleet of cloud servers running production workloads. The agent observes real-time utilization, latency, cost, and SLA status — then takes infrastructure actions (provision, terminate, resize, migrate) to minimize cost while keeping everything healthy.

### Why this matters for RL/GRPO training

- **Multi-objective optimization:** Cost and performance are in tension. There is no single "right answer" — only tradeoffs. This tests whether an agent can find the efficient frontier.
- **Sequential decision-making with ordering constraints:** You can't terminate a server before migrating its workloads. You can't migrate without free capacity. Actions must be sequenced correctly.
- **Partial credit everywhere:** The reward function gives continuous signal, not binary pass/fail. An agent that reduces cost from $2.40 to $0.90 scores much higher than one that does nothing — even if $0.90 is still over the $0.50 budget.
- **Deterministic grading:** Same actions always produce the same score. No randomness, no hidden state. Pure reasoning benchmark.

---

## 2. Observation Space

Every step, the agent receives a full snapshot of the infrastructure:

| Field | Type | Description |
|---|---|---|
| `servers` | `List[ServerInfo]` | All active server instances in the fleet (see sub-fields below) |
| `workloads` | `List[WorkloadInfo]` | All workloads that must be kept running (see sub-fields below) |
| `total_cost_per_hour` | `float` | Sum of all running server costs in USD/hr |
| `budget_per_hour` | `float` | Target budget the agent should optimize toward |
| `sla_violations` | `int` | Count of workloads currently breaching their latency SLA |
| `unassigned_workloads` | `int` | Count of workloads not running on any server (DOWN) |
| `step_number` | `int` | Current step in the episode (0-indexed) |
| `max_steps` | `int` | Maximum steps allowed before episode ends |
| `done` | `bool` | True if the episode is over |
| `message` | `str` | Human-readable feedback about the last action |

### ServerInfo sub-fields

| Field | Type | Description |
|---|---|---|
| `server_id` | `str` | Unique identifier (e.g. `"srv-001"`) |
| `instance_type` | `str` | Tier: `nano` / `micro` / `small` / `medium` / `large` / `xlarge` |
| `cpu_cores` | `int` | Number of vCPU cores |
| `ram_gb` | `int` | RAM in GB |
| `cpu_utilization` | `float` | Current CPU usage, 0.0–1.0 |
| `ram_utilization` | `float` | Current RAM usage, 0.0–1.0 |
| `cost_per_hour` | `float` | Hourly cost in USD |
| `assigned_workloads` | `List[str]` | Workload IDs running on this server |

### WorkloadInfo sub-fields

| Field | Type | Description |
|---|---|---|
| `workload_id` | `str` | Unique identifier (e.g. `"wl-api"`) |
| `name` | `str` | Human-readable name (e.g. `"API Gateway"`) |
| `required_cpu` | `float` | Minimum CPU cores needed |
| `required_ram` | `float` | Minimum RAM in GB needed |
| `current_latency_ms` | `float` | Current p95 latency in milliseconds |
| `sla_latency_ms` | `float` | Maximum acceptable latency before SLA breach |
| `is_critical` | `bool` | If true, SLA breach carries 2× penalty weight |
| `assigned_server` | `str \| null` | Server ID, or `null` if unassigned (DOWN) |

### Instance Type Catalog

| Type | CPU | RAM | Cost/hr |
|---|---|---|---|
| `nano` | 1 | 1 GB | $0.05 |
| `micro` | 1 | 2 GB | $0.10 |
| `small` | 2 | 4 GB | $0.20 |
| `medium` | 4 | 8 GB | $0.40 |
| `large` | 8 | 16 GB | $0.80 |
| `xlarge` | 16 | 32 GB | $1.60 |

---

## 3. Action Space

Each step, the agent submits exactly one action:

| Field | Type | Options | Description |
|---|---|---|---|
| `action_type` | `str` | `provision`, `terminate`, `resize`, `migrate`, `noop` | The infrastructure operation to perform |
| `server_id` | `str \| null` | Any valid server ID | Target server (required for `terminate`, `resize`) |
| `instance_type` | `str \| null` | `nano`, `micro`, `small`, `medium`, `large`, `xlarge` | Instance tier (required for `provision`, `resize`) |
| `workload_id` | `str \| null` | Any valid workload ID | Workload to move (required for `migrate`) |
| `target_server_id` | `str \| null` | Any valid server ID | Destination server (required for `migrate`) |

### Action semantics

| Action | Effect | Risk |
|---|---|---|
| `provision` | Creates a new server (empty, no workloads) | Increases cost without benefit if nothing is migrated to it |
| `terminate` | Destroys a server permanently | **Orphans all workloads** on it — they become unassigned (DOWN). Incurs +0.3 penalty. |
| `resize` | Changes a server's instance type in-place | Keeps workloads assigned. May cause SLA breach if downsized too aggressively. |
| `migrate` | Moves a workload from its current server to another | Target server must exist. Also works for assigning orphaned workloads. |
| `noop` | Does nothing | Wastes a step. 3+ consecutive noops incur escalating penalty. |

---

## 4. Reward Function

The reward is computed at every step as a composite of three components:

```
score = (cost_efficiency × 0.50) + (performance_score × 0.40) − (penalty × 0.10)
```

Clamped to **[0.0, 1.0]**. Fully deterministic — same actions always produce the same score.

### Cost Efficiency (50% weight)

Rewards lower cost, not just being "under budget":

| Total Cost vs Budget | Cost Efficiency Score |
|---|---|
| Zero cost (theoretical) | 1.00 |
| 50% of budget | 0.90 |
| At budget | 0.80 |
| 2× budget | 0.40 |
| 3× budget or more | 0.00 |

**Example:** Budget is $0.50/hr. Agent reduces cost from $2.40 to $0.80.
Cost efficiency = 0.80 − 0.40 × (0.80 − 0.50) / 0.50 = **0.56**. Partial credit for progress.

### Performance Score (40% weight)

Weighted fraction of workloads that are both assigned to a server AND meeting their SLA:

- Each healthy workload contributes its weight (1.0 for normal, **2.0 for critical**)
- Unassigned workloads and SLA-breaching workloads contribute 0

**Example:** 4 workloads (2 critical). API Gateway (critical) is breaching SLA, rest are healthy.
Weights: 2+2+1+1 = 6. Healthy: 2+1+1 = 4. Performance = 4/6 = **0.667**.

### Penalty (10% weight, deducted)

| Penalty Source | Amount | Rationale |
|---|---|---|
| Repeated identical action | +0.10 per repeat | Prevents loops — agents must make progress |
| Destructive termination (server with workloads) | +0.30 per occurrence | Orphaning workloads is the worst mistake |
| 3+ trailing noops | +0.05 per noop beyond 2 | Prevents agents from giving up early |
| 2+ provisions without any migration | +0.10 | Prevents mindless server spawning |

Penalty is clamped to 1.0 maximum.

---

## 5. Tasks

### Task 1: Single Server Rightsizing

| | |
|---|---|
| **Difficulty** | 🟢 Easy |
| **Max Steps** | 5 |
| **Budget** | $0.20/hr |
| **Objective** | Resize a massively over-provisioned xlarge server ($1.60/hr) to match a lightweight workload that only needs 1 CPU / 2 GB RAM |

**Starting state:** 1 xlarge server running 1 workload ("Company Website"). The server is using 6% of its capacity — pure waste.

**What a good agent does:** Resize `srv-001` from `xlarge` to `micro` in a single action. Cost drops from $1.60 to $0.10. SLA stays met. Score: ~0.85.

**What a bad agent does:** Terminates the server (orphans the workload), provisions a new one but forgets to migrate, or does nothing.

---

### Task 2: Multi-Service Consolidation

| | |
|---|---|
| **Difficulty** | 🟡 Medium |
| **Max Steps** | 10 |
| **Budget** | $0.50/hr |
| **Objective** | Consolidate 4 workloads from 4 oversized servers ($2.40/hr total) onto fewer right-sized servers |

**Starting state:** 4 servers (2 medium, 2 large), each running a single workload. Combined utilization is ~15%. Two workloads are **critical** (API Gateway, Database Replica).

**What a good agent does:** Migrates all workloads onto 1 large server ($0.80/hr) and terminates the other 3. This requires correct ordering: migrate first, terminate second. Score: ~0.68.

**What a bad agent does:** Over-consolidates onto a medium server (causes SLA breaches on critical workloads), terminates servers before migrating (orphans workloads), or provisions new servers without using them.

**The tradeoff:** A single large server ($0.80) keeps all SLAs met but exceeds the $0.50 budget. A medium+small combo ($0.60) is cheaper but risks overload. The agent must balance cost vs. reliability.

---

### Task 3: Fleet Chaos Triage

| | |
|---|---|
| **Difficulty** | 🔴 Hard |
| **Max Steps** | 15 |
| **Budget** | $0.80/hr |
| **Objective** | Fix a broken fleet — 3 SLA breaches, 1 orphaned workload, 1 idle server, massive over-provisioning |

**Starting state (the mess):**

| Server | Type | Cost | Status |
|---|---|---|---|
| srv-001 | medium | $0.40 | **OVERLOADED** — 2 critical workloads, both breaching SLA |
| srv-002 | large | $0.80 | OK but oversized — 1 workload using 25% capacity |
| srv-003 | nano | $0.05 | **UNDERSIZED** — workload needs 2 GB RAM, server has 1 GB |
| srv-004 | medium | $0.40 | **COMPLETELY IDLE** — no workloads, pure waste |
| srv-005 | nano | $0.05 | OK — right-sized for metrics collector |

Total cost: $1.70/hr. Budget: $0.80/hr. 3 SLA violations. 1 orphaned workload (`wl-ml-jobs`).

**What a good agent does:** (requires 8–12 carefully ordered actions)
1. Migrate `wl-api` off overloaded `srv-001` to idle `srv-004`
2. Migrate `wl-search` off undersized `srv-003` to a capable server
3. Assign orphaned `wl-ml-jobs` to a server with capacity
4. Terminate or resize waste servers
5. Right-size remaining fleet to cut cost

**Why it's hard:** Every action has dependencies. You can't terminate before migrating. You can't migrate to a full server. The total resource demand (10.5 CPU, 15.5 GB) exceeds the cheapest single-server option. Even GPT-4/Claude struggle to plan the full sequence correctly.

---

## 6. Local Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-dir>

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install fastapi uvicorn pydantic openai requests

# 4. Start the environment server
python env.py
# Server starts at http://localhost:7860

# 5. Test it (in another terminal)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

---

## 7. Docker Setup

```bash
# Build the image
docker build -t openenv-devops .

# Run the container
docker run -p 7860:7860 openenv-devops

# Verify it's running
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy"}'
```

The container runs as a non-root user on port 7860 (HuggingFace Spaces default).

---

## 8. Running Inference

The baseline agent uses an LLM to play all 3 tasks:

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export ENV_URL="http://localhost:7860"  # or your HF Space URL

# Make sure the environment server is running, then:
python inference.py
```

**Windows PowerShell:**

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
$env:HF_TOKEN="hf_your_token_here"
$env:ENV_URL="http://localhost:7860"
python inference.py
```

Typical runtime: **2–3 minutes** on 2 vCPU / 8 GB RAM (well under the 20-minute limit).

---

## 9. Baseline Scores

| Task | Difficulty | Max Steps | Budget | Baseline Score | Random Agent Score |
|---|---|---|---|---|---|
| Single Server Rightsizing | 🟢 Easy | 5 | $0.20/hr | 0.80–0.85 | ~0.30–0.40 |
| Multi-Service Consolidation | 🟡 Medium | 10 | $0.50/hr | 0.40–0.55 | ~0.15–0.25 |
| Fleet Chaos Triage | 🔴 Hard | 15 | $0.80/hr | 0.15–0.30 | ~0.05–0.15 |

- **Baseline** = LLM agent (e.g. Llama-3.1-8B-Instruct) using the system prompt in `inference.py`
- **Random agent** = uniformly random valid actions
- All scores are deterministic: same actions → same score, always

---

## 10. OpenEnv Validation

```bash
# Install the OpenEnv CLI
pip install openenv-core

# Validate the environment spec
openenv validate

# Test that the server responds correctly
curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:7860/reset
# Expected: 200
```

---

## API Reference

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/reset` | POST | `{"task_id": "easy\|medium\|hard"}` | Initial `Observation` |
| `/step` | POST | `Action` JSON | `{observation, reward, done, info}` |
| `/state` | GET | — | Full current state including action history |

---

## Technical Stack

- **Python 3.11** + **FastAPI** + **Pydantic v2** + **uvicorn**
- **OpenAI client** for LLM inference (compatible with any OpenAI-format API)
- **Zero ML dependencies** in the environment itself
- **Fully deterministic** grading — no randomness anywhere
- **Single-file environment** (`env.py`) — models, tasks, grading, and server in one place
