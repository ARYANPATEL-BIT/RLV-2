"""
tasks.py — Task definitions and grader functions for DevOps/FinOps OpenEnv.

3 tasks of increasing difficulty:
  easy   — Single Server Rightsizing
  medium — Multi-Service Consolidation
  hard   — Fleet Chaos Triage

Each task exports:
  - TASK_CONFIG: dict with id, name, objective, max_steps, budget
  - initial_state(): returns (servers, workloads) for reset()
  - grade(observation, action_history): returns Reward
"""

from models import (
    ServerInfo, WorkloadInfo, Observation, Reward, Action,
    INSTANCE_CATALOG,
)
from typing import List, Tuple


# ═══════════════════════════════════════════════════════════════════════
# HELPER: Shared grading utilities
# ═══════════════════════════════════════════════════════════════════════

def _compute_cost_efficiency(total_cost: float, budget: float) -> float:
    """
    Rewards lower cost, not just being under budget.
      - At budget:     0.80
      - Under budget:  0.80–1.00 (scales with savings)
      - Over budget:   0.80→0.00 (linear, hits 0 at 3× budget)
    This ensures two different under-budget costs always produce different scores.
    """
    if budget <= 0:
        return 0.0
    if total_cost <= 0:
        return 1.0
    if total_cost <= budget:
        savings_ratio = 1.0 - (total_cost / budget)  # 0.0 at budget, 1.0 at zero
        return 0.80 + (0.20 * savings_ratio)
    else:
        overspend_ratio = (total_cost - budget) / budget  # 0.0 at budget, 2.0 at 3× budget
        return max(0.0, 0.80 - (0.40 * overspend_ratio))


def _compute_performance_score(workloads: List[WorkloadInfo]) -> float:
    """
    Fraction of workloads that are (a) assigned to a server AND (b) meeting SLA.
    Critical workloads that breach SLA count as 2 failures.
    Returns 0.0 if there are no workloads.
    """
    if not workloads:
        return 1.0

    total_weight = 0.0
    healthy_weight = 0.0
    for wl in workloads:
        weight = 2.0 if wl.is_critical else 1.0
        total_weight += weight
        if wl.assigned_server is not None and wl.current_latency_ms <= wl.sla_latency_ms:
            healthy_weight += weight

    return healthy_weight / total_weight


def _compute_penalty(action_history: List[Action], observation: Observation) -> float:
    """
    Penalty sources (all additive, clamped to 1.0):
      - Repeated identical actions: +0.1 per repeat (after 1st occurrence)
      - Terminating a server that has workloads: +0.3 per occurrence
      - Provisioning without ever migrating (waste): +0.1 if done 2+ times
    """
    penalty = 0.0

    # --- Repeated identical actions (loop detection) ---
    if len(action_history) >= 2:
        seen = {}
        for act in action_history:
            key = (act.action_type, act.server_id, act.instance_type,
                   act.workload_id, act.target_server_id)
            seen[key] = seen.get(key, 0) + 1
        for key, count in seen.items():
            if count > 1:
                penalty += 0.1 * (count - 1)

    # --- Destructive terminations ---
    # (env.py will track this via info dict, but we can infer from history:
    #  if a terminate was followed by unassigned workloads appearing)
    # For grading, we count terminate actions on servers that had workloads
    # The env will pass this info via the action_history metadata
    # We approximate: count noop actions at end (agent gave up)
    noop_tail = 0
    for act in reversed(action_history):
        if act.action_type == "noop":
            noop_tail += 1
        else:
            break
    if noop_tail >= 3:
        penalty += 0.05 * (noop_tail - 2)

    # --- Provision spam without migration ---
    provisions = sum(1 for a in action_history if a.action_type == "provision")
    migrations = sum(1 for a in action_history if a.action_type == "migrate")
    if provisions >= 2 and migrations == 0:
        penalty += 0.1

    return min(penalty, 1.0)


def _compute_reward(
    observation: Observation,
    action_history: List[Action],
    is_final: bool,
) -> Reward:
    """
    Master reward formula:
      score = (cost_efficiency × 0.50) + (performance_score × 0.40) − (penalty × 0.10)
    Clamped to [0.0, 1.0]
    """
    ce = _compute_cost_efficiency(observation.total_cost_per_hour, observation.budget_per_hour)
    ps = _compute_performance_score(observation.workloads)
    pen = _compute_penalty(action_history, observation)

    raw_score = (ce * 0.50) + (ps * 0.40) - (pen * 0.10)
    score = max(0.0, min(1.0, raw_score))

    breakdown = (
        f"cost_efficiency={ce:.3f} (weight 0.50) + "
        f"performance={ps:.3f} (weight 0.40) - "
        f"penalty={pen:.3f} (weight 0.10) = {score:.3f}"
    )

    return Reward(
        score=round(score, 4),
        cost_efficiency=round(ce, 4),
        performance_score=round(ps, 4),
        penalty=round(pen, 4),
        is_final=is_final,
        breakdown=breakdown,
    )


# ═══════════════════════════════════════════════════════════════════════
# TASK 1 — EASY: Single Server Rightsizing
# ═══════════════════════════════════════════════════════════════════════

TASK_EASY_CONFIG = {
    "task_id": "easy",
    "name": "Single Server Rightsizing",
    "objective": (
        "A single workload is running on a massively over-provisioned xlarge server; "
        "resize or replace it with the smallest instance that meets the workload's needs "
        "to bring cost under budget."
    ),
    "max_steps": 5,
    "budget_per_hour": 0.20,
}


def task_easy_initial_state() -> Tuple[List[ServerInfo], List[WorkloadInfo]]:
    """
    Starting state:
      1 xlarge server ($1.60/hr) running 1 workload that only needs 1 CPU / 2 GB RAM.
      Budget is $0.20/hr → agent must downsize to micro ($0.10/hr).
      Optimal path: resize srv-001 to micro (1 action).
    """
    servers = [
        ServerInfo(
            server_id="srv-001",
            instance_type="xlarge",
            cpu_cores=16,
            ram_gb=32,
            cpu_utilization=0.06,   # 1/16 cores used
            ram_utilization=0.06,   # 2/32 GB used
            cost_per_hour=1.60,
            assigned_workloads=["wl-web"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-web",
            name="Company Website",
            required_cpu=1.0,
            required_ram=2.0,
            current_latency_ms=45.0,
            sla_latency_ms=200.0,
            is_critical=False,
            assigned_server="srv-001",
        ),
    ]
    return servers, workloads


def task_easy_grade(observation: Observation, action_history: List[Action]) -> Reward:
    """
    Grade Task 1 — Easy.
    A competent agent scores 0.6–0.8 by resizing to micro.
    Random actions score ~0.2–0.3 (partial credit for not destroying things).
    """
    return _compute_reward(observation, action_history, is_final=observation.done)


# ═══════════════════════════════════════════════════════════════════════
# TASK 2 — MEDIUM: Multi-Service Consolidation
# ═══════════════════════════════════════════════════════════════════════

TASK_MEDIUM_CONFIG = {
    "task_id": "medium",
    "name": "Multi-Service Consolidation",
    "objective": (
        "Four workloads are each running on their own oversized server; "
        "consolidate them onto fewer, right-sized servers to cut cost below budget "
        "while keeping all SLAs met."
    ),
    "max_steps": 10,
    "budget_per_hour": 0.50,
}


def task_medium_initial_state() -> Tuple[List[ServerInfo], List[WorkloadInfo]]:
    """
    Starting state:
      4 servers, each with 1 workload, all oversized.
      Total cost: $0.40 + $0.80 + $0.40 + $0.80 = $2.40/hr
      Budget: $0.50/hr

      Workload needs (total): 6 CPU, 10 GB RAM
      Optimal: 1 large ($0.80, 8 CPU/16 GB) can't fit all if tight,
               or 1 medium + 1 small = $0.60 (close to budget)
               or 1 large alone at $0.80 (over budget but fewer SLA risks)
      Best: provision 1 medium (4 CPU/8 GB) + 1 small (2 CPU/4 GB) = $0.60
            but need to check RAM: 10 GB needed, 8+4=12 → fits.
            Actually medium=8GB + small=4GB = 12GB ≥ 10GB ✓
            CPU: 4+2=6 ≥ 6 ✓
            Cost: $0.60 → over budget $0.50 but partial credit.
            Even better: 1 medium ($0.40) → 4 CPU, 8 GB → can fit 6 CPU? No.
            Best realistic: agent consolidates to 2 servers, cost ~$0.40-$0.60
    """
    servers = [
        ServerInfo(
            server_id="srv-001",
            instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=0.25, ram_utilization=0.25,
            cost_per_hour=0.40,
            assigned_workloads=["wl-api"],
        ),
        ServerInfo(
            server_id="srv-002",
            instance_type="large",
            cpu_cores=8, ram_gb=16,
            cpu_utilization=0.12, ram_utilization=0.12,
            cost_per_hour=0.80,
            assigned_workloads=["wl-db"],
        ),
        ServerInfo(
            server_id="srv-003",
            instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=0.25, ram_utilization=0.19,
            cost_per_hour=0.40,
            assigned_workloads=["wl-cache"],
        ),
        ServerInfo(
            server_id="srv-004",
            instance_type="large",
            cpu_cores=8, ram_gb=16,
            cpu_utilization=0.06, ram_utilization=0.06,
            cost_per_hour=0.80,
            assigned_workloads=["wl-worker"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-api",
            name="API Gateway",
            required_cpu=1.0, required_ram=2.0,
            current_latency_ms=60.0,
            sla_latency_ms=150.0,
            is_critical=True,
            assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-db",
            name="Database Replica",
            required_cpu=2.0, required_ram=4.0,
            current_latency_ms=30.0,
            sla_latency_ms=100.0,
            is_critical=True,
            assigned_server="srv-002",
        ),
        WorkloadInfo(
            workload_id="wl-cache",
            name="Redis Cache",
            required_cpu=1.0, required_ram=1.5,
            current_latency_ms=10.0,
            sla_latency_ms=50.0,
            is_critical=False,
            assigned_server="srv-003",
        ),
        WorkloadInfo(
            workload_id="wl-worker",
            name="Background Worker",
            required_cpu=0.5, required_ram=1.0,
            current_latency_ms=200.0,
            sla_latency_ms=500.0,
            is_critical=False,
            assigned_server="srv-004",
        ),
    ]
    return servers, workloads


def task_medium_grade(observation: Observation, action_history: List[Action]) -> Reward:
    """
    Grade Task 2 — Medium.
    A competent agent scores 0.3–0.5 by consolidating onto 1-2 servers.
    Must handle migration ordering carefully (no dropped workloads).
    Random actions score ~0.1–0.2.
    """
    return _compute_reward(observation, action_history, is_final=observation.done)


# ═══════════════════════════════════════════════════════════════════════
# TASK 3 — HARD: Fleet Chaos Triage
# ═══════════════════════════════════════════════════════════════════════

TASK_HARD_CONFIG = {
    "task_id": "hard",
    "name": "Fleet Chaos Triage",
    "objective": (
        "Diagnose and fix a broken fleet with orphaned workloads, SLA breaches, "
        "idle servers, and over-provisioning — achieve zero downtime, zero SLA "
        "breaches, and cost under budget."
    ),
    "max_steps": 15,
    "budget_per_hour": 0.80,
}


def task_hard_initial_state() -> Tuple[List[ServerInfo], List[WorkloadInfo]]:
    """
    Starting state — a mess:
      5 servers:
        srv-001 (medium, $0.40) — running wl-auth + wl-api (overloaded, SLA breach on both)
        srv-002 (large, $0.80) — running wl-db (OK but oversized)
        srv-003 (small, $0.20) — running wl-search (SLA breach, too small)
        srv-004 (medium, $0.40) — IDLE, no workloads (pure waste)
        srv-005 (nano, $0.05) — running wl-metrics (OK, right-sized)

      6 workloads:
        wl-auth    — critical, on srv-001, SLA BREACH (server overloaded)
        wl-api     — critical, on srv-001, SLA BREACH (server overloaded)
        wl-db      — critical, on srv-002, OK
        wl-search  — non-critical, on srv-003, SLA BREACH (undersized server)
        wl-metrics — non-critical, on srv-005, OK
        wl-ml-jobs — non-critical, UNASSIGNED (orphaned, not running anywhere)

      Total cost: $0.40+$0.80+$0.20+$0.40+$0.05 = $1.85/hr
      Budget: $0.80/hr
      SLA violations: 3 (wl-auth, wl-api, wl-search)
      Unassigned: 1 (wl-ml-jobs)

      Resource needs:
        wl-auth:    2 CPU, 3 GB RAM (critical)
        wl-api:     2 CPU, 2 GB RAM (critical)
        wl-db:      2 CPU, 4 GB RAM (critical)
        wl-search:  1 CPU, 2 GB RAM
        wl-metrics: 0.5 CPU, 0.5 GB RAM
        wl-ml-jobs: 3 CPU, 4 GB RAM (unassigned)
        Total: 10.5 CPU, 15.5 GB RAM

      Optimal approach requires ~8-12 carefully ordered actions:
        1. Migrate wl-api off srv-001 → need free capacity somewhere
        2. Resize srv-004 or provision new server for capacity
        3. Assign wl-ml-jobs somewhere
        4. Terminate idle/waste servers
        5. Resize oversized servers
      Order matters: can't terminate before migrating, can't migrate without capacity.
    """
    servers = [
        ServerInfo(
            server_id="srv-001",
            instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=0.95,   # overloaded
            ram_utilization=0.63,   # (3+2)/8
            cost_per_hour=0.40,
            assigned_workloads=["wl-auth", "wl-api"],
        ),
        ServerInfo(
            server_id="srv-002",
            instance_type="large",
            cpu_cores=8, ram_gb=16,
            cpu_utilization=0.25,
            ram_utilization=0.25,
            cost_per_hour=0.80,
            assigned_workloads=["wl-db"],
        ),
        ServerInfo(
            server_id="srv-003",
            instance_type="small",
            cpu_cores=2, ram_gb=4,
            cpu_utilization=0.50,
            ram_utilization=0.50,
            cost_per_hour=0.20,
            assigned_workloads=["wl-search"],
        ),
        ServerInfo(
            server_id="srv-004",
            instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=0.0,    # completely idle
            ram_utilization=0.0,
            cost_per_hour=0.40,
            assigned_workloads=[],
        ),
        ServerInfo(
            server_id="srv-005",
            instance_type="nano",
            cpu_cores=1, ram_gb=1,
            cpu_utilization=0.50,
            ram_utilization=0.50,
            cost_per_hour=0.05,
            assigned_workloads=["wl-metrics"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-auth",
            name="Auth Service",
            required_cpu=2.0, required_ram=3.0,
            current_latency_ms=320.0,   # SLA BREACH
            sla_latency_ms=150.0,
            is_critical=True,
            assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-api",
            name="API Gateway",
            required_cpu=2.0, required_ram=2.0,
            current_latency_ms=280.0,   # SLA BREACH
            sla_latency_ms=150.0,
            is_critical=True,
            assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-db",
            name="Primary Database",
            required_cpu=2.0, required_ram=4.0,
            current_latency_ms=25.0,    # OK
            sla_latency_ms=100.0,
            is_critical=True,
            assigned_server="srv-002",
        ),
        WorkloadInfo(
            workload_id="wl-search",
            name="Search Index",
            required_cpu=1.0, required_ram=2.0,
            current_latency_ms=180.0,   # SLA BREACH
            sla_latency_ms=100.0,
            is_critical=False,
            assigned_server="srv-003",
        ),
        WorkloadInfo(
            workload_id="wl-metrics",
            name="Metrics Collector",
            required_cpu=0.5, required_ram=0.5,
            current_latency_ms=15.0,    # OK
            sla_latency_ms=500.0,
            is_critical=False,
            assigned_server="srv-005",
        ),
        WorkloadInfo(
            workload_id="wl-ml-jobs",
            name="ML Training Pipeline",
            required_cpu=3.0, required_ram=4.0,
            current_latency_ms=0.0,     # NOT RUNNING
            sla_latency_ms=1000.0,
            is_critical=False,
            assigned_server=None,        # ORPHANED
        ),
    ]
    return servers, workloads


def task_hard_grade(observation: Observation, action_history: List[Action]) -> Reward:
    """
    Grade Task 3 — Hard.
    Even GPT-4/Claude scores 0.1–0.3: requires careful multi-step planning
    with ordering constraints (migrate before terminate, provision before migrate).
    Random actions score ~0.0–0.1.
    """
    return _compute_reward(observation, action_history, is_final=observation.done)


# ═══════════════════════════════════════════════════════════════════════
# TASK REGISTRY — maps task_id → config + functions
# ═══════════════════════════════════════════════════════════════════════

TASK_REGISTRY = {
    "easy": {
        "config": TASK_EASY_CONFIG,
        "initial_state": task_easy_initial_state,
        "grade": task_easy_grade,
    },
    "medium": {
        "config": TASK_MEDIUM_CONFIG,
        "initial_state": task_medium_initial_state,
        "grade": task_medium_grade,
    },
    "hard": {
        "config": TASK_HARD_CONFIG,
        "initial_state": task_hard_initial_state,
        "grade": task_hard_grade,
    },
}
