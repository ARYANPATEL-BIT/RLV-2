---
title: OpenEnv DevOps FinOps Cloud Optimizer
emoji: ☁️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ☁️ OpenEnv: DevOps/FinOps Cloud Optimizer

**An AI agent environment where models learn to optimize cloud infrastructure cost vs. performance under realistic operational constraints.**

---

## 1. Environment Overview & Motivation

Cloud infrastructure is often the largest variable cost for software companies. FinOps (Financial Operations) is the discipline of maximizing cloud value, but today, these decisions rely heavily on engineers manually triaging dashboards.

This OpenEnv environment challenges AI agents to automate FinOps. It simulates a fleet of cloud servers running production workloads across a **4-dimensional resource model** (CPU, RAM, disk IOPS, network bandwidth). The agent observes real-time utilization, latency, cost, and SLA status, and takes infrastructure actions to minimize cost without sacrificing reliability.

### Key Environment Mechanics (v2.0)
*   **4-Dimensional Resources:** Servers and workloads are defined by CPU, RAM, Disk IOPS, and Network. A bottleneck in *any* dimension causes SLA breaches.
*   **11 Instance Types:** Ranges from standard computing to specialized tiers (compute-optimized, memory-optimized) and **Spot Instances** (70% cheaper but deterministically evicted, penalizing greedy agents).
*   **Cascading Failures:** Workloads have dependency graphs (e.g., API depends on Database). If a dependency goes down, the reliant workload suffers latency penalties.
*   **Traffic Spikes:** Seed-based deterministic demand fluctuations test an agent's ability to provision adaptively.

### Why this is a great benchmark for RL/GRPO:
*   **Multi-objective Optimization:** The reward function places cost and performance in tension. Continuous partial credit accurately measures reasoning capability.
*   **Complex Sequential Decisions:** Agents must migrate workloads before terminating servers, requiring strong planning.
*   **100% Deterministic:** Pure stateless reasoning benchmark. The exact same actions always yield the exact same score.

---

## 2. Observation Space

At every step, the agent receives a comprehensive snapshot of the cloud infrastructure:

### Global State
*   `total_cost_per_hour`: Sum of running servers in USD/hr.
*   `budget_per_hour`: Target FinOps budget constraint.
*   `sla_violations`: Number of workloads breaching latency SLA.
*   `unassigned_workloads`: Number of workloads currently completely DOWN.

### Servers (`List[ServerInfo]`)
A list of all active servers, including attributes like:
*   `server_id` and `instance_type`.
*   Capacity metrics: `cpu_cores`, `ram_gb`, `disk_iops`, `network_gbps`.
*   Current utilization: `cpu_utilization`, `ram_utilization`, `disk_utilization`, `network_utilization` (uncapped, can exceed 100% when overloaded).
*   Status flags: `is_spot`, `is_overloaded`.

### Workloads (`List[WorkloadInfo]`)
The apps that must stay online, including attributes like:
*   `workload_id` and semantic `name`.
*   Demand metrics minimums (CPU, RAM, Disk, Net).
*   SLA Tracking: `current_latency_ms` vs. `sla_latency_ms`.
*   Status flags: `is_critical` (2x penalty if SLA breached), `dependencies` (cascading failures).

---

## 3. Action Space

Every step, the agent outputs exactly one JSON action specifying the infrastructure operation.

| Action Type | Required Parameters | Description & Risks |
| :--- | :--- | :--- |
| `provision` | `instance_type` | Spawns a new server. (Increases cost, requires migration to utilize). |
| `terminate` | `server_id` | Permanently destroys a server. (**Danger:** Orphans all workloads to it, applying a massive 0.50 penalty). |
| `resize` | `server_id`, `instance_type`| Changes the tier in-place. (Requires ensuring workloads still fit). |
| `migrate` | `workload_id`, `target_server_id` | Moves a workload. (Target server may become overloaded). |
| `noop` | *None* | Do nothing. (Wastes a step. 3 consecutive noops incur a penalty). |

---

## 4. Task Descriptions & Difficulty Levels

The environment offers three statically defined tasks of increasing complexity, plus procedural generation.

### 🟢 Target 1: Single Server Rightsizing (Easy)
*   **Budget:** $0.25/hr | **Max Steps:** 5
*   **Objective:** Two workloads are running on a massively over-provisioned `xlarge` server costing $1.60/hr. The agent must resize to the smallest instance that supports both without breaching SLAs.

### 🟡 Target 2: Multi-Service Consolidation (Medium)
*   **Budget:** $0.50/hr | **Max Steps:** 10
*   **Objective:** Four workloads are isolated on their own oversized servers (total $2.40/hr). The agent must migrate workloads to consolidate them onto fewer servers, navigating a cascading dependency from the API layer to the database. 

### 🔴 Target 3: Fleet Chaos Triage (Hard)
*   **Budget:** $1.20/hr | **Max Steps:** 15
*   **Objective:** The agent enters a completely broken fleet. There are 3 immediate SLA breaches, 1 orphaned workload offline, cascading database failures, and gross over-provisioning elsewhere. Fixing this requires 8-12 perfectly sequenced migration and resize actions.

### 🎲 Procedural Generation (Unlimited)
By passing `task_id="random"` and a `seed`, the environment dynamically generates a fleet configuration, varying server capacity, dependencies, and SLAs, preventing models from simply memorizing the deterministic tasks.

---

## 5. Setup and Usage Instructions

The environment has zero ML dependencies (FastAPI only) and acts as a localized standalone REST API.

### Local Setup (Python 3.11+)
```bash
# 1. Clone & prepare environment
git clone <repo-url> && cd <repo-dir>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run unit tests to validate grading determinism
python test_env.py

# 3. Start the environment server (Runs on port 7860)
python env.py
```

### Docker Setup
```bash
docker build -t openenv-devops .
docker run -p 7860:7860 openenv-devops
```

### Testing the Interactive Dashboard
The repository includes a localized HTML Dashboard to visualize action executions.
1. Create a session: `curl -X POST http://localhost:7860/reset -d '{"task_id": "easy"}'`
2. Open your browser directly to: `http://localhost:7860/dashboard`

### Running Inference Script (LLM Evaluation)
To run the provided `inference.py` script against the environment:
```bash
export API_BASE_URL="https://router.huggingface.co/v1" # Or any OpenAI-format base URL
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_huggingface_token" 

python inference.py
```

---

## 6. Baseline Performance Scores

Scores are clamped perfectly to `[0.0, 1.0]` based on: `(Cost Efficiency × 0.40) + (SLA Performance × 0.35) - (Penalty × 0.25)`.

*Because grading is purely deterministic, an identical action trajectory will consistently produce these results.*

| Task | Difficulty | Max Steps | Budget | Standard Baseline (Llama 3.1 8B) | Random Baseline (Random Action Picks) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rightsizing** | 🟢 Easy | 5 | $0.25/hr | **0.65 – 0.75** | ~0.15 – 0.30 |
| **Consolidation**| 🟡 Medium | 10 | $0.50/hr | **0.40 – 0.55** | ~0.10 – 0.20 |
| **Chaos Triage** | 🔴 Hard | 15 | $1.20/hr | **0.25 – 0.40** | ~0.05 – 0.15 |

*Note: For a model to achieve perfect scores (0.90+) on Chaos Triage, it must perfectly orchestrate the sequence of bottleneck identification, migration target validation, dependency ordering, and capacity planning. This is explicitly designed to be structurally rigorous.*
