"""
env.py — DevOps/FinOps OpenEnv Environment v2.0

All-in-one: Pydantic models, latency simulation, spot eviction,
cascading failures, traffic spikes, 3 fixed tasks + procedural
generation, grading, session management, FastAPI endpoints,
and live dashboard.

Stack: Python 3.11, FastAPI, Pydantic v2, uvicorn
Port:  7860 (HuggingFace Spaces default)
"""

from __future__ import annotations

import copy

import random
import threading
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════
# INSTANCE CATALOG — 11 types: 6 standard + 3 specialized + 2 spot
# ═══════════════════════════════════════════════════════════════════════

class InstanceSpec(TypedDict):
    cpu: int
    ram: int
    disk_iops: int
    net_gbps: float
    cost: float
    is_spot: bool


INSTANCE_CATALOG: Dict[str, InstanceSpec] = {
    "nano":        {"cpu": 1,  "ram": 1,  "disk_iops": 1000,  "net_gbps": 0.5,  "cost": 0.05, "is_spot": False},
    "micro":       {"cpu": 1,  "ram": 2,  "disk_iops": 2000,  "net_gbps": 0.5,  "cost": 0.10, "is_spot": False},
    "small":       {"cpu": 2,  "ram": 4,  "disk_iops": 3000,  "net_gbps": 1.0,  "cost": 0.20, "is_spot": False},
    "medium":      {"cpu": 4,  "ram": 8,  "disk_iops": 5000,  "net_gbps": 2.0,  "cost": 0.40, "is_spot": False},
    "large":       {"cpu": 8,  "ram": 16, "disk_iops": 10000, "net_gbps": 5.0,  "cost": 0.80, "is_spot": False},
    "xlarge":      {"cpu": 16, "ram": 32, "disk_iops": 20000, "net_gbps": 10.0, "cost": 1.60, "is_spot": False},
    "compute-opt": {"cpu": 8,  "ram": 8,  "disk_iops": 5000,  "net_gbps": 5.0,  "cost": 0.60, "is_spot": False},
    "memory-opt":  {"cpu": 4,  "ram": 32, "disk_iops": 10000, "net_gbps": 2.0,  "cost": 0.70, "is_spot": False},
    "storage-opt": {"cpu": 4,  "ram": 8,  "disk_iops": 30000, "net_gbps": 2.0,  "cost": 0.55, "is_spot": False},
    "spot-medium": {"cpu": 4,  "ram": 8,  "disk_iops": 5000,  "net_gbps": 2.0,  "cost": 0.12, "is_spot": True},
    "spot-large":  {"cpu": 8,  "ram": 16, "disk_iops": 10000, "net_gbps": 5.0,  "cost": 0.24, "is_spot": True},
}

ALL_INSTANCE_TYPES = Literal[
    "nano", "micro", "small", "medium", "large", "xlarge",
    "compute-opt", "memory-opt", "storage-opt", "spot-medium", "spot-large",
]


# ═══════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════

class ServerInfo(BaseModel):
    server_id: str
    instance_type: str
    cpu_cores: int
    ram_gb: int
    disk_iops: int = 1000
    network_gbps: float = 0.5
    cpu_utilization: float = 0.0
    ram_utilization: float = 0.0
    disk_utilization: float = 0.0
    network_utilization: float = 0.0
    cost_per_hour: float = Field(ge=0.0)
    assigned_workloads: List[str] = Field(default_factory=list)
    is_spot: bool = False
    is_overloaded: bool = False


class WorkloadInfo(BaseModel):
    workload_id: str
    name: str
    required_cpu: float = Field(gt=0.0)
    required_ram: float = Field(gt=0.0)
    required_disk_iops: int = 500
    required_net_gbps: float = 0.1
    current_latency_ms: float = Field(ge=0.0)
    sla_latency_ms: float = Field(gt=0.0)
    is_critical: bool
    assigned_server: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    servers: List[ServerInfo]
    workloads: List[WorkloadInfo]
    total_cost_per_hour: float = Field(ge=0.0)
    budget_per_hour: float = Field(gt=0.0)
    sla_violations: int = Field(ge=0)
    unassigned_workloads: int = Field(ge=0)
    step_number: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    done: bool
    message: str
    traffic_multiplier_active: bool = False
    spot_eviction_occurred: bool = False


class Action(BaseModel):
    action_type: Literal["provision", "terminate", "resize", "migrate", "noop"]
    server_id: Optional[str] = None
    instance_type: Optional[str] = None
    workload_id: Optional[str] = None
    target_server_id: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    cost_efficiency: float = Field(ge=0.0, le=1.0)
    performance_score: float = Field(ge=0.0, le=1.0)
    penalty: float = Field(ge=0.0, le=1.0)
    is_final: bool
    breakdown: str


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════
# LATENCY SIMULATION — 4-dimensional resource model
# ═══════════════════════════════════════════════════════════════════════

def _get_traffic_multiplier(seed: Optional[int], step: int, workload_id: str) -> float:
    """Deterministic traffic multiplier. Range [0.85, 1.45]."""
    if seed is None:
        return 1.0
    h = hash((seed, step, workload_id)) & 0xFFFFFFFF
    return 0.85 + (h % 1000) / 1000 * 0.6


def _simulate_latency(
    workload: WorkloadInfo,
    server_cpu: int, server_ram: int, server_disk: int, server_net: float,
    workloads_on_server: List[WorkloadInfo],
    traffic_mult: float = 1.0,
) -> float:
    """
    Deterministic latency from 4-dimensional resource pressure.
    Uses max of all resource ratios as the bottleneck dimension.
    Context-switching overhead: +5% per additional co-located workload.
    """
    total_cpu = sum(w.required_cpu * traffic_mult for w in workloads_on_server)
    total_ram = sum(w.required_ram * traffic_mult for w in workloads_on_server)
    total_disk = sum(w.required_disk_iops * traffic_mult for w in workloads_on_server)
    total_net = sum(w.required_net_gbps * traffic_mult for w in workloads_on_server)

    cpu_r = total_cpu / server_cpu if server_cpu > 0 else 10.0
    ram_r = total_ram / server_ram if server_ram > 0 else 10.0
    disk_r = total_disk / server_disk if server_disk > 0 else 10.0
    net_r = total_net / server_net if server_net > 0 else 10.0

    base_util = max(cpu_r, ram_r, disk_r, net_r)
    overhead = 0.05 * max(0, len(workloads_on_server) - 1)
    utilization = base_util + overhead
    sla = workload.sla_latency_ms

    if utilization <= 0.5:
        return round(sla * 0.2, 1)
    elif utilization <= 1.0:
        t = (utilization - 0.5) / 0.5
        return round(sla * (0.2 + t * 0.55), 1)
    else:
        excess = utilization - 1.0
        return round(sla * (1.1 + excess * 2.0), 1)


def _apply_cascading_failures(workloads: List[WorkloadInfo]) -> None:
    """If a dependency is DOWN or breaching SLA, inflate dependent's latency by 1.5x."""
    wl_map = {w.workload_id: w for w in workloads}
    for wl in workloads:
        for dep_id in wl.dependencies:
            dep = wl_map.get(dep_id)
            if dep and (dep.assigned_server is None or dep.current_latency_ms > dep.sla_latency_ms):
                wl.current_latency_ms = round(wl.current_latency_ms * 1.5, 1)


# ═══════════════════════════════════════════════════════════════════════
# GRADING — weights: cost 0.40, performance 0.35, penalty 0.25
# ═══════════════════════════════════════════════════════════════════════

WEIGHT_COST = 0.40
WEIGHT_PERF = 0.35
WEIGHT_PEN = 0.25


def _compute_cost_efficiency(total_cost: float, budget: float) -> float:
    if budget <= 0:
        return 0.0
    if total_cost <= 0:
        return 1.0
    if total_cost <= budget:
        return 0.80 + 0.20 * (1.0 - total_cost / budget)
    else:
        overspend = (total_cost - budget) / budget
        return max(0.0, 0.80 - 0.40 * overspend)


def _compute_performance_score(workloads: List[WorkloadInfo]) -> float:
    if not workloads:
        return 1.0
    total_w = 0.0
    healthy_w = 0.0
    for wl in workloads:
        w = 2.0 if wl.is_critical else 1.0
        total_w += w
        if wl.assigned_server is not None and wl.current_latency_ms <= wl.sla_latency_ms:
            healthy_w += w
    return healthy_w / total_w


def _compute_penalty(action_history: List[Action], destructive_count: int) -> float:
    penalty = 0.0
    non_noop_actions = [a for a in action_history if a.action_type != "noop"]
    # Repeated identical NON-NOOP actions (noops excluded — they're valid idle)
    if len(non_noop_actions) >= 2:
        seen: Dict[tuple, int] = {}
        for act in non_noop_actions:
            key = (act.action_type, act.server_id, act.instance_type,
                   act.workload_id, act.target_server_id)
            seen[key] = seen.get(key, 0) + 1
        for key, count in seen.items():
            if count > 1:
                penalty += 0.15 * (count - 1)
    # Consecutive identical NON-NOOP actions (loop detection — extra penalty)
    if len(action_history) >= 3:
        consecutive = 1
        for i in range(1, len(action_history)):
            prev = action_history[i - 1]
            curr = action_history[i]
            same = (prev.action_type == curr.action_type and prev.server_id == curr.server_id
                    and prev.instance_type == curr.instance_type
                    and prev.workload_id == curr.workload_id
                    and prev.target_server_id == curr.target_server_id)
            if same and curr.action_type != "noop":
                consecutive += 1
            else:
                consecutive = 1
            if consecutive >= 3:
                penalty += 0.1  # extra penalty per loop iteration beyond 2
    # Destructive terminations — 0.50 each (severe)
    penalty += 0.50 * destructive_count
    # All-noop penalty: if agent did NOTHING useful, penalize idleness
    # But if agent took useful actions then stopped, trailing noops are fine
    if len(non_noop_actions) == 0 and len(action_history) >= 3:
        penalty += 0.05 * (len(action_history) - 2)
    # Provision spam
    provisions = sum(1 for a in non_noop_actions if a.action_type == "provision")
    migrations = sum(1 for a in non_noop_actions if a.action_type == "migrate")
    if provisions >= 2 and migrations == 0:
        penalty += 0.1
    return min(penalty, 1.0)


def _compute_reward(
    observation: Observation, action_history: List[Action],
    destructive_count: int, is_final: bool,
) -> Reward:
    ce = _compute_cost_efficiency(observation.total_cost_per_hour, observation.budget_per_hour)
    ps = _compute_performance_score(observation.workloads)
    # If ALL workloads are unhealthy, cost savings alone shouldn't score well
    if ps == 0.0:
        ce = 0.0
    pen = _compute_penalty(action_history, destructive_count)
    raw = (ce * WEIGHT_COST) + (ps * WEIGHT_PERF) - (pen * WEIGHT_PEN)
    score = max(0.0, min(1.0, raw))
    breakdown = (
        f"cost_eff={ce:.3f}×{WEIGHT_COST} + perf={ps:.3f}×{WEIGHT_PERF}"
        f" - penalty={pen:.3f}×{WEIGHT_PEN} = {score:.4f}"
        f" | NOTE: if perf=0 then cost_eff is zeroed (all workloads unhealthy)"
    )
    return Reward(
        score=round(score, 4), cost_efficiency=round(ce, 4),
        performance_score=round(ps, 4), penalty=round(pen, 4),
        is_final=is_final, breakdown=breakdown,
    )


# ═══════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

class TaskConfig(TypedDict):
    name: str
    objective: str
    max_steps: int
    budget_per_hour: float


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "easy": {
        "name": "Single Server Rightsizing",
        "objective": (
            "Two workloads run on a massively over-provisioned xlarge server ($1.60/hr). "
            "Resize to the smallest instance that fits BOTH workloads' CPU+RAM+disk+network "
            "needs without breaching SLAs. Budget: $0.25/hr."
        ),
        "max_steps": 5,
        "budget_per_hour": 0.25,
    },
    "medium": {
        "name": "Multi-Service Consolidation",
        "objective": (
            "Consolidate 4 workloads from 4 oversized servers ($2.40/hr) onto fewer "
            "right-sized servers. wl-api depends on wl-db — if DB breaches, API cascades. "
            "Budget: $0.50/hr."
        ),
        "max_steps": 10,
        "budget_per_hour": 0.50,
    },
    "hard": {
        "name": "Fleet Chaos Triage",
        "objective": (
            "Fix a broken fleet: 3 SLA breaches, 1 orphaned workload, 1 idle server, "
            "cascading failures, and over-provisioning. wl-api depends on wl-auth; "
            "wl-search depends on wl-db. Budget: $1.20/hr."
        ),
        "max_steps": 15,
        "budget_per_hour": 1.20,
    },
}


def _build_easy_state():
    servers = [
        ServerInfo(
            server_id="srv-001", instance_type="xlarge",
            cpu_cores=16, ram_gb=32, disk_iops=20000, network_gbps=10.0,
            cpu_utilization=0.09, ram_utilization=0.08,
            cost_per_hour=1.60, assigned_workloads=["wl-web", "wl-api"],
            is_spot=False, is_overloaded=False,
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-web", name="Company Website",
            required_cpu=0.5, required_ram=1.0, required_disk_iops=500, required_net_gbps=0.1,
            current_latency_ms=40.0, sla_latency_ms=200.0,
            is_critical=False, assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-api", name="API Gateway",
            required_cpu=1.0, required_ram=1.5, required_disk_iops=800, required_net_gbps=0.3,
            current_latency_ms=30.0, sla_latency_ms=150.0,
            is_critical=True, assigned_server="srv-001",
        ),
    ]
    return servers, workloads


def _build_medium_state():
    servers = [
        ServerInfo(
            server_id="srv-001", instance_type="medium",
            cpu_cores=4, ram_gb=8, disk_iops=5000, network_gbps=2.0,
            cpu_utilization=0.25, ram_utilization=0.25,
            cost_per_hour=0.40, assigned_workloads=["wl-api"],
        ),
        ServerInfo(
            server_id="srv-002", instance_type="large",
            cpu_cores=8, ram_gb=16, disk_iops=10000, network_gbps=5.0,
            cpu_utilization=0.25, ram_utilization=0.25,
            cost_per_hour=0.80, assigned_workloads=["wl-db"],
        ),
        ServerInfo(
            server_id="srv-003", instance_type="medium",
            cpu_cores=4, ram_gb=8, disk_iops=5000, network_gbps=2.0,
            cpu_utilization=0.25, ram_utilization=0.19,
            cost_per_hour=0.40, assigned_workloads=["wl-cache"],
        ),
        ServerInfo(
            server_id="srv-004", instance_type="large",
            cpu_cores=8, ram_gb=16, disk_iops=10000, network_gbps=5.0,
            cpu_utilization=0.06, ram_utilization=0.06,
            cost_per_hour=0.80, assigned_workloads=["wl-worker"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-api", name="API Gateway",
            required_cpu=1.0, required_ram=2.0, required_disk_iops=800, required_net_gbps=0.3,
            current_latency_ms=30.0, sla_latency_ms=150.0,
            is_critical=True, assigned_server="srv-001",
            dependencies=["wl-db"],
        ),
        WorkloadInfo(
            workload_id="wl-db", name="Database Replica",
            required_cpu=2.0, required_ram=4.0, required_disk_iops=2000, required_net_gbps=0.5,
            current_latency_ms=20.0, sla_latency_ms=100.0,
            is_critical=True, assigned_server="srv-002",
        ),
        WorkloadInfo(
            workload_id="wl-cache", name="Redis Cache",
            required_cpu=1.0, required_ram=1.5, required_disk_iops=500, required_net_gbps=0.2,
            current_latency_ms=10.0, sla_latency_ms=50.0,
            is_critical=False, assigned_server="srv-003",
        ),
        WorkloadInfo(
            workload_id="wl-worker", name="Background Worker",
            required_cpu=0.5, required_ram=1.0, required_disk_iops=300, required_net_gbps=0.1,
            current_latency_ms=100.0, sla_latency_ms=500.0,
            is_critical=False, assigned_server="srv-004",
        ),
    ]
    return servers, workloads


def _build_hard_state():
    servers = [
        ServerInfo(
            server_id="srv-001", instance_type="medium",
            cpu_cores=4, ram_gb=8, disk_iops=5000, network_gbps=2.0,
            cpu_utilization=1.0, ram_utilization=0.63,
            cost_per_hour=0.40, assigned_workloads=["wl-auth", "wl-api"],
        ),
        ServerInfo(
            server_id="srv-002", instance_type="large",
            cpu_cores=8, ram_gb=16, disk_iops=10000, network_gbps=5.0,
            cpu_utilization=0.25, ram_utilization=0.25,
            cost_per_hour=0.80, assigned_workloads=["wl-db"],
        ),
        ServerInfo(
            server_id="srv-003", instance_type="nano",
            cpu_cores=1, ram_gb=1, disk_iops=1000, network_gbps=0.5,
            cpu_utilization=1.0, ram_utilization=1.0,
            cost_per_hour=0.05, assigned_workloads=["wl-search"],
        ),
        ServerInfo(
            server_id="srv-004", instance_type="medium",
            cpu_cores=4, ram_gb=8, disk_iops=5000, network_gbps=2.0,
            cpu_utilization=0.0, ram_utilization=0.0,
            cost_per_hour=0.40, assigned_workloads=[],
        ),
        ServerInfo(
            server_id="srv-005", instance_type="nano",
            cpu_cores=1, ram_gb=1, disk_iops=1000, network_gbps=0.5,
            cpu_utilization=0.50, ram_utilization=0.50,
            cost_per_hour=0.05, assigned_workloads=["wl-metrics"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-auth", name="Auth Service",
            required_cpu=2.0, required_ram=3.0, required_disk_iops=600, required_net_gbps=0.2,
            current_latency_ms=180.0, sla_latency_ms=150.0,
            is_critical=True, assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-api", name="API Gateway",
            required_cpu=2.0, required_ram=2.0, required_disk_iops=800, required_net_gbps=0.3,
            current_latency_ms=180.0, sla_latency_ms=150.0,
            is_critical=True, assigned_server="srv-001",
            dependencies=["wl-auth"],
        ),
        WorkloadInfo(
            workload_id="wl-db", name="Primary Database",
            required_cpu=2.0, required_ram=4.0, required_disk_iops=2000, required_net_gbps=0.5,
            current_latency_ms=20.0, sla_latency_ms=100.0,
            is_critical=True, assigned_server="srv-002",
        ),
        WorkloadInfo(
            workload_id="wl-search", name="Search Index",
            required_cpu=1.0, required_ram=2.0, required_disk_iops=1000, required_net_gbps=0.2,
            current_latency_ms=310.0, sla_latency_ms=100.0,
            is_critical=False, assigned_server="srv-003",
            dependencies=["wl-db"],
        ),
        WorkloadInfo(
            workload_id="wl-metrics", name="Metrics Collector",
            required_cpu=0.5, required_ram=0.5, required_disk_iops=200, required_net_gbps=0.1,
            current_latency_ms=100.0, sla_latency_ms=500.0,
            is_critical=False, assigned_server="srv-005",
        ),
        WorkloadInfo(
            workload_id="wl-ml-jobs", name="ML Training Pipeline",
            required_cpu=3.0, required_ram=4.0, required_disk_iops=3000, required_net_gbps=0.5,
            current_latency_ms=0.0, sla_latency_ms=1000.0,
            is_critical=False, assigned_server=None,
        ),
    ]
    return servers, workloads


TASK_BUILDERS = {
    "easy": _build_easy_state,
    "medium": _build_medium_state,
    "hard": _build_hard_state,
}


# ═══════════════════════════════════════════════════════════════════════
# PROCEDURAL TASK GENERATOR
# ═══════════════════════════════════════════════════════════════════════

_WL_NAMES = [
    "API Gateway", "Auth Service", "Primary Database", "Search Index",
    "Cache Layer", "Background Worker", "ML Pipeline", "Web Frontend",
    "Logging Service", "Message Queue", "Payment Processor", "Notification Hub",
]


def _build_random_state(seed: int):
    rng = random.Random(seed)
    non_spot = [k for k, v in INSTANCE_CATALOG.items() if not v["is_spot"]]
    num_srv = rng.randint(3, 7)
    num_wl = rng.randint(4, 8)

    servers = []
    for i in range(num_srv):
        itype = rng.choice(non_spot)
        spec = INSTANCE_CATALOG[itype]
        servers.append(ServerInfo(
            server_id=f"srv-{i+1:03d}", instance_type=itype,
            cpu_cores=spec["cpu"], ram_gb=spec["ram"],
            disk_iops=spec["disk_iops"], network_gbps=spec["net_gbps"],
            cost_per_hour=spec["cost"], is_spot=spec["is_spot"],
        ))

    workloads = []
    for i in range(num_wl):
        cpu = rng.choice([0.5, 1.0, 1.5, 2.0, 3.0])
        ram = rng.choice([0.5, 1.0, 2.0, 3.0, 4.0])
        disk = rng.choice([300, 500, 1000, 2000, 3000])
        net = rng.choice([0.1, 0.2, 0.3, 0.5, 1.0])
        sla = rng.choice([50, 100, 150, 200, 500, 1000])
        crit = rng.random() < 0.3
        srv = rng.choice(servers).server_id if rng.random() < 0.85 else None
        deps = []
        if i > 0 and rng.random() < 0.2:
            deps = [workloads[rng.randint(0, i - 1)].workload_id]
        workloads.append(WorkloadInfo(
            workload_id=f"wl-{i+1:03d}", name=_WL_NAMES[i % len(_WL_NAMES)],
            required_cpu=cpu, required_ram=ram,
            required_disk_iops=disk, required_net_gbps=net,
            current_latency_ms=0.0, sla_latency_ms=float(sla),
            is_critical=crit, assigned_server=srv, dependencies=deps,
        ))

    # Solvability check: ensure total demand doesn't exceed 2x total capacity
    total_cpu_demand = sum(w.required_cpu for w in workloads)
    total_ram_demand = sum(w.required_ram for w in workloads)
    total_cpu_cap = sum(s.cpu_cores for s in servers)
    total_ram_cap = sum(s.ram_gb for s in servers)
    if total_cpu_demand > 2 * total_cpu_cap or total_ram_demand > 2 * total_ram_cap:
        # Add a large server to make the task feasible
        spec = INSTANCE_CATALOG["large"]
        servers.append(ServerInfo(
            server_id=f"srv-{len(servers)+1:03d}", instance_type="large",
            cpu_cores=spec["cpu"], ram_gb=spec["ram"],
            disk_iops=spec["disk_iops"], network_gbps=spec["net_gbps"],
            cost_per_hour=spec["cost"], is_spot=False,
        ))

    total_cost = sum(s.cost_per_hour for s in servers)
    budget = round(total_cost * rng.uniform(0.4, 0.65), 2)
    max_steps = rng.choice([8, 10, 12, 15])
    return servers, workloads, budget, max_steps


# ═══════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS
# ═══════════════════════════════════════════════════════════════════════

class CloudEnvironment:
    def __init__(self):
        self._lock = threading.Lock()
        self.servers: List[ServerInfo] = []
        self.workloads: List[WorkloadInfo] = []
        self.task_id: str = "easy"
        self.budget: float = 0.25
        self.max_steps: int = 5
        self.step_number: int = 0
        self.done: bool = True
        self.action_history: List[Action] = []
        self.destructive_count: int = 0
        self._server_counter: int = 0
        self._last_message: str = "[INFO] No episode running. Call /reset first."
        self.seed: Optional[int] = None
        self._spot_eviction_this_step: bool = False

    def _next_server_id(self) -> str:
        self._server_counter += 1
        return f"srv-{self._server_counter:03d}"

    def _find_server(self, sid: str) -> Optional[ServerInfo]:
        return next((s for s in self.servers if s.server_id == sid), None)

    def _find_workload(self, wid: str) -> Optional[WorkloadInfo]:
        return next((w for w in self.workloads if w.workload_id == wid), None)

    def _workloads_on_server(self, sid: str) -> List[WorkloadInfo]:
        return [w for w in self.workloads if w.assigned_server == sid]

    def _check_spot_evictions(self) -> List[str]:
        """Deterministic spot eviction at steps 3, 7, 12."""
        if self.step_number not in (3, 7, 12):
            return []
        evicted_names = []
        for srv in list(self.servers):
            if not srv.is_spot:
                continue
            h = hash((self.seed or 0, self.step_number, srv.server_id)) & 0xFFFFFFFF
            if h % 100 < 40:
                orphaned = self._workloads_on_server(srv.server_id)
                for wl in orphaned:
                    wl.assigned_server = None
                    wl.current_latency_ms = 0.0
                self.servers = [s for s in self.servers if s.server_id != srv.server_id]
                evicted_names.append(srv.server_id)
        return evicted_names

    def _recalculate_state(self):
        traffic_active = self.seed is not None
        for srv in self.servers:
            wls = self._workloads_on_server(srv.server_id)
            tc, tr, td, tn = 0.0, 0.0, 0.0, 0.0
            for w in wls:
                m = _get_traffic_multiplier(self.seed, self.step_number, w.workload_id) if traffic_active else 1.0
                tc += w.required_cpu * m
                tr += w.required_ram * m
                td += w.required_disk_iops * m
                tn += w.required_net_gbps * m
            srv.cpu_utilization = round(tc / srv.cpu_cores, 4) if srv.cpu_cores > 0 else 0.0
            srv.ram_utilization = round(tr / srv.ram_gb, 4) if srv.ram_gb > 0 else 0.0
            srv.disk_utilization = round(td / srv.disk_iops, 4) if srv.disk_iops > 0 else 0.0
            srv.network_utilization = round(tn / srv.network_gbps, 4) if srv.network_gbps > 0 else 0.0
            srv.is_overloaded = max(srv.cpu_utilization, srv.ram_utilization,
                                    srv.disk_utilization, srv.network_utilization) > 1.0
            srv.assigned_workloads = [w.workload_id for w in wls]

        for wl in self.workloads:
            if wl.assigned_server is None:
                wl.current_latency_ms = 0.0
                continue
            srv = self._find_server(wl.assigned_server)
            if srv is None:
                wl.assigned_server = None
                wl.current_latency_ms = 0.0
                continue
            wls_on = self._workloads_on_server(srv.server_id)
            # Per-workload traffic multiplier: use the workload's own ID for the hash
            mult = _get_traffic_multiplier(self.seed, self.step_number, wl.workload_id) if self.seed else 1.0
            wl.current_latency_ms = _simulate_latency(
                wl, srv.cpu_cores, srv.ram_gb, srv.disk_iops, srv.network_gbps,
                wls_on, mult,
            )

        _apply_cascading_failures(self.workloads)

    def _build_observation(self) -> Observation:
        total_cost = sum(s.cost_per_hour for s in self.servers)
        sla_v = sum(1 for w in self.workloads
                    if w.assigned_server is not None and w.current_latency_ms > w.sla_latency_ms)
        unassigned = sum(1 for w in self.workloads if w.assigned_server is None)
        return Observation(
            servers=copy.deepcopy(self.servers),
            workloads=copy.deepcopy(self.workloads),
            total_cost_per_hour=round(total_cost, 4),
            budget_per_hour=self.budget,
            sla_violations=sla_v, unassigned_workloads=unassigned,
            step_number=self.step_number, max_steps=self.max_steps,
            done=self.done, message=self._last_message,
            traffic_multiplier_active=self.seed is not None,
            spot_eviction_occurred=self._spot_eviction_this_step,
        )

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> Observation:
        if task_id == "random":
            s = seed if seed is not None else int(time.time())
            servers, workloads, budget, max_steps = _build_random_state(s)
            self.seed = s
            self.task_id = task_id
            self.budget = budget
            self.max_steps = max_steps
            self.servers = servers
            self.workloads = workloads
        elif task_id in TASK_CONFIGS:
            config = TASK_CONFIGS[task_id]
            servers, workloads = TASK_BUILDERS[task_id]()
            self.task_id = task_id
            self.budget = config["budget_per_hour"]
            self.max_steps = config["max_steps"]
            self.servers = servers
            self.workloads = workloads
            self.seed = seed
        else:
            raise ValueError(f"Unknown task_id '{task_id}'. Use: {list(TASK_CONFIGS.keys()) + ['random']}")

        self.step_number = 0
        self.done = False
        self.action_history = []
        self.destructive_count = 0
        self._spot_eviction_this_step = False
        max_id = 0
        for s in self.servers:
            num = int(s.server_id.split("-")[1])
            if num > max_id:
                max_id = num
        self._server_counter = max_id
        self._recalculate_state()
        obj = TASK_CONFIGS.get(task_id, {"name": f"Random (seed={self.seed})", "objective": "Optimize the fleet."})
        self._last_message = f"[INFO] Episode started: {obj['name']}. {obj.get('objective', '')}"
        return self._build_observation()

    def step(self, action: Action) -> dict:
        if self.done:
            obs = self._build_observation()
            reward = _compute_reward(obs, self.action_history, self.destructive_count, True)
            return {"observation": obs, "reward": reward, "done": True,
                    "info": {"error": "Episode is done. Call /reset to start a new one."}}

        # Spot eviction at start of step
        self._spot_eviction_this_step = False
        evicted = self._check_spot_evictions()
        if evicted:
            self._spot_eviction_this_step = True
            self._last_message = f"[EVICTION] Spot instances evicted: {evicted}. Workloads orphaned."
            self._recalculate_state()

        self.action_history.append(action)
        msg = self._execute_action(action)
        self._last_message = msg
        self._recalculate_state()
        self.step_number += 1
        if self.step_number >= self.max_steps:
            self.done = True
        obs = self._build_observation()
        reward = _compute_reward(obs, self.action_history, self.destructive_count, self.done)
        return {"observation": obs, "reward": reward, "done": self.done,
                "info": {"step": self.step_number, "action_result": msg}}

    def _execute_action(self, action: Action) -> str:
        atype = action.action_type
        if atype == "noop":
            return "[OK] No operation performed."

        elif atype == "provision":
            if action.instance_type is None:
                return "[ERROR:MISSING_PARAM] provision requires instance_type."
            if action.instance_type not in INSTANCE_CATALOG:
                return f"[ERROR:INVALID_TYPE] Unknown instance_type '{action.instance_type}'."
            spec = INSTANCE_CATALOG[action.instance_type]
            new_id = self._next_server_id()
            self.servers.append(ServerInfo(
                server_id=new_id, instance_type=action.instance_type,
                cpu_cores=spec["cpu"], ram_gb=spec["ram"],
                disk_iops=spec["disk_iops"], network_gbps=spec["net_gbps"],
                cost_per_hour=spec["cost"], is_spot=spec["is_spot"],
            ))
            spot_warn = " ⚠️ SPOT: may be evicted at steps 3/7/12." if spec["is_spot"] else ""
            return f"[OK] Provisioned {action.instance_type} server {new_id} (${spec['cost']}/hr).{spot_warn}"

        elif atype == "terminate":
            if action.server_id is None:
                return "[ERROR:MISSING_PARAM] terminate requires server_id."
            srv = self._find_server(action.server_id)
            if srv is None:
                return f"[ERROR:NOT_FOUND] Server {action.server_id} not found."
            orphaned = self._workloads_on_server(srv.server_id)
            if orphaned:
                self.destructive_count += 1
                for wl in orphaned:
                    wl.assigned_server = None
                    wl.current_latency_ms = 0.0
            self.servers = [s for s in self.servers if s.server_id != action.server_id]
            if orphaned:
                names = [w.workload_id for w in orphaned]
                return (f"[DESTRUCTIVE] Terminated {action.server_id}. "
                        f"{len(orphaned)} workloads orphaned: {names}. Heavy penalty applied.")
            return f"[OK] Terminated idle server {action.server_id}."

        elif atype == "resize":
            if action.server_id is None or action.instance_type is None:
                return "[ERROR:MISSING_PARAM] resize requires server_id and instance_type."
            if action.instance_type not in INSTANCE_CATALOG:
                return f"[ERROR:INVALID_TYPE] Unknown instance_type '{action.instance_type}'."
            srv = self._find_server(action.server_id)
            if srv is None:
                return f"[ERROR:NOT_FOUND] Server {action.server_id} not found."
            if srv.instance_type == action.instance_type:
                return f"[WARN:NO_CHANGE] Server {action.server_id} is already {action.instance_type}."
            old_type = srv.instance_type
            spec = INSTANCE_CATALOG[action.instance_type]
            srv.instance_type = action.instance_type
            srv.cpu_cores = spec["cpu"]
            srv.ram_gb = spec["ram"]
            srv.disk_iops = spec["disk_iops"]
            srv.network_gbps = spec["net_gbps"]
            srv.cost_per_hour = spec["cost"]
            srv.is_spot = spec["is_spot"]
            # Capacity warning
            wls = self._workloads_on_server(srv.server_id)
            tcpu = sum(w.required_cpu for w in wls)
            tram = sum(w.required_ram for w in wls)
            warn = ""
            if tcpu > spec["cpu"] or tram > spec["ram"]:
                warn = f" ⚠️ WARNING: workloads need {tcpu}cpu/{tram}GB but {action.instance_type} has {spec['cpu']}cpu/{spec['ram']}GB. SLA breach likely."
            return f"[OK] Resized {action.server_id} from {old_type} to {action.instance_type} (${spec['cost']}/hr).{warn}"

        elif atype == "migrate":
            if action.workload_id is None or action.target_server_id is None:
                return "[ERROR:MISSING_PARAM] migrate requires workload_id and target_server_id."
            wl = self._find_workload(action.workload_id)
            if wl is None:
                return f"[ERROR:NOT_FOUND] Workload {action.workload_id} not found."
            target = self._find_server(action.target_server_id)
            if target is None:
                return f"[ERROR:NOT_FOUND] Target server {action.target_server_id} not found."
            if wl.assigned_server == action.target_server_id:
                return f"[WARN:NO_CHANGE] {action.workload_id} is already on {action.target_server_id}."
            old_server = wl.assigned_server
            if old_server:
                src = self._find_server(old_server)
                if src:
                    src.assigned_workloads = [w for w in src.assigned_workloads if w != action.workload_id]
            wl.assigned_server = action.target_server_id
            target.assigned_workloads.append(action.workload_id)
            origin = old_server or "unassigned"
            # Capacity warning
            wls_on_target = self._workloads_on_server(action.target_server_id)
            tcpu = sum(w.required_cpu for w in wls_on_target)
            tram = sum(w.required_ram for w in wls_on_target)
            warn = ""
            if tcpu > target.cpu_cores or tram > target.ram_gb:
                warn = f" ⚠️ WARNING: target now needs {tcpu}cpu/{tram}GB but has {target.cpu_cores}cpu/{target.ram_gb}GB. Overload likely."
            return f"[OK] Migrated {action.workload_id} from {origin} to {action.target_server_id}.{warn}"

        return f"[ERROR:UNKNOWN] Unknown action_type '{atype}'."

    def get_state(self) -> dict:
        obs = self._build_observation()
        return {
            "task_id": self.task_id, "observation": obs,
            "action_history": [a.model_dump() for a in self.action_history],
            "destructive_terminations": self.destructive_count,
        }


# ═══════════════════════════════════════════════════════════════════════
# SESSION MANAGER
# ═══════════════════════════════════════════════════════════════════════

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, CloudEnvironment] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._latest: Optional[str] = None

    def create(self) -> str:
        with self._lock:
            self._cleanup()
            sid = uuid.uuid4().hex[:8]
            self._sessions[sid] = CloudEnvironment()
            self._timestamps[sid] = time.time()
            self._latest = sid
            return sid

    def get(self, sid: Optional[str]) -> Optional[CloudEnvironment]:
        with self._lock:
            if sid is None:
                sid = self._latest
            if sid and sid in self._sessions:
                self._timestamps[sid] = time.time()
                return self._sessions[sid]
            return None

    def latest_id(self) -> Optional[str]:
        return self._latest

    def _cleanup(self):
        now = time.time()
        expired = [s for s, t in self._timestamps.items() if now - t > 3600]
        for s in expired:
            del self._sessions[s]
            del self._timestamps[s]


# ═══════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DevOps/FinOps OpenEnv",
    description="AI agent environment for cloud infrastructure cost vs. performance optimization.",
    version="2.0.0",
)
sessions = SessionManager()


@app.get("/")
def health_check():
    return {"status": "ok", "environment": "devops-finops-cloud-optimizer", "version": "2.0.0"}


@app.post("/reset")
def reset_endpoint(body: ResetRequest = ResetRequest()):
    sid = sessions.create()
    env = sessions.get(sid)
    try:
        obs = env.reset(body.task_id, body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": sid, **obs.model_dump()}


@app.post("/step")
def step_endpoint(action: Action, session_id: Optional[str] = Query(None)):
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id. Call /reset first.")
    with env._lock:
        if env.done and env.step_number == 0:
            raise HTTPException(status_code=400, detail="No episode running. Call /reset first.")
        result = env.step(action)
        return {
            "observation": result["observation"].model_dump(),
            "reward": result["reward"].model_dump(),
            "done": result["done"],
            "info": result["info"],
        }


@app.get("/state")
def state_endpoint(session_id: Optional[str] = Query(None)):
    env = sessions.get(session_id)
    if env is None:
        raise HTTPException(status_code=400, detail="Invalid or missing session_id. Call /reset first.")
    with env._lock:
        state = env.get_state()
        return {
            "session_id": session_id or sessions.latest_id(),
            "task_id": state["task_id"],
            "observation": state["observation"].model_dump(),
            "action_history": state["action_history"],
            "destructive_terminations": state["destructive_terminations"],
        }


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

_DASHBOARD_CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Inter,system-ui,sans-serif;background:#0a0a1a;color:#e0e0e0;padding:20px}
h1{text-align:center;font-size:1.8rem;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
.sub{text-align:center;color:#888;margin-bottom:20px;font-size:0.85rem}
.metrics{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:24px}
.metric{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:14px 22px;min-width:140px;text-align:center;backdrop-filter:blur(10px)}
.metric .label{font-size:0.75rem;color:#aaa;text-transform:uppercase;letter-spacing:1px}
.metric .value{display:block;font-size:1.5rem;font-weight:700;margin-top:4px}
.ok{color:#34d399}.warn{color:#fbbf24}.bad{color:#f87171}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-bottom:24px}
.card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:14px;padding:16px;transition:transform .2s}
.card:hover{transform:translateY(-2px);border-color:rgba(96,165,250,0.3)}
.card h3{font-size:0.95rem;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.spot-badge{background:#f59e0b;color:#000;font-size:0.6rem;padding:2px 6px;border-radius:4px;font-weight:700}
.overload-badge{background:#ef4444;font-size:0.6rem;padding:2px 6px;border-radius:4px;font-weight:700}
.bar-container{height:8px;background:rgba(255,255,255,0.1);border-radius:4px;margin:4px 0 8px;overflow:hidden}
.bar{height:100%;border-radius:4px;transition:width .5s ease}
.bar-ok{background:linear-gradient(90deg,#34d399,#6ee7b7)}
.bar-warn{background:linear-gradient(90deg,#fbbf24,#f59e0b)}
.bar-bad{background:linear-gradient(90deg,#f87171,#ef4444)}
.bar-label{font-size:0.7rem;color:#aaa;display:flex;justify-content:space-between}
.wl-section h2{font-size:1.1rem;margin-bottom:12px;color:#a78bfa}
.wl-item{display:flex;align-items:center;gap:10px;padding:8px 12px;background:rgba(255,255,255,0.03);border-radius:8px;margin-bottom:6px}
.wl-status{font-size:0.65rem;padding:3px 8px;border-radius:4px;font-weight:700;min-width:55px;text-align:center}
.st-ok{background:#065f46;color:#6ee7b7}.st-breach{background:#7f1d1d;color:#fca5a5}.st-down{background:#374151;color:#9ca3af}
.wl-name{font-weight:600;flex:1}.wl-detail{font-size:0.75rem;color:#888}
.dep{font-size:0.65rem;color:#a78bfa;margin-left:4px}
"""


def _bar_class(pct: float) -> str:
    if pct <= 60:
        return "bar-ok"
    elif pct <= 85:
        return "bar-warn"
    return "bar-bad"


def _render_dashboard(env: CloudEnvironment, sid: str) -> str:
    obs = env._build_observation()
    cost_class = "ok" if obs.total_cost_per_hour <= obs.budget_per_hour else ("warn" if obs.total_cost_per_hour <= obs.budget_per_hour * 1.5 else "bad")
    sla_class = "ok" if obs.sla_violations == 0 else "bad"
    un_class = "ok" if obs.unassigned_workloads == 0 else "bad"

    server_cards = ""
    for s in obs.servers:
        badges = ""
        if s.is_spot:
            badges += '<span class="spot-badge">⚡SPOT</span>'
        if s.is_overloaded:
            badges += '<span class="overload-badge">🔥OVERLOADED</span>'
        cpu_pct = min(s.cpu_utilization * 100, 200)
        ram_pct = min(s.ram_utilization * 100, 200)
        disk_pct = min(s.disk_utilization * 100, 200)
        net_pct = min(s.network_utilization * 100, 200)

        def bar(label, pct):
            w = min(pct, 100)
            return f'''<div class="bar-label"><span>{label}</span><span>{pct:.0f}%</span></div>
<div class="bar-container"><div class="bar {_bar_class(pct)}" style="width:{w}%"></div></div>'''

        server_cards += f'''<div class="card">
<h3>{s.server_id} ({s.instance_type}) ${s.cost_per_hour}/hr {badges}</h3>
<div style="font-size:0.75rem;color:#888;margin-bottom:8px">{s.cpu_cores}cpu / {s.ram_gb}GB / {s.disk_iops}iops / {s.network_gbps}Gbps</div>
{bar("CPU", cpu_pct)}{bar("RAM", ram_pct)}{bar("Disk", disk_pct)}{bar("Net", net_pct)}
<div style="font-size:0.75rem;color:#ccc;margin-top:4px">Workloads: {', '.join(s.assigned_workloads) or 'none'}</div>
</div>'''

    wl_items = ""
    for w in obs.workloads:
        if w.assigned_server is None:
            st, stc = "DOWN", "st-down"
        elif w.current_latency_ms > w.sla_latency_ms:
            st, stc = "BREACH", "st-breach"
        else:
            st, stc = "OK", "st-ok"
        crit = " [CRIT]" if w.is_critical else ""
        dep_str = f'<span class="dep">→ depends on {", ".join(w.dependencies)}</span>' if w.dependencies else ""
        wl_items += f'''<div class="wl-item">
<span class="wl-status {stc}">{st}</span>
<span class="wl-name">{w.workload_id} — {w.name}{crit}{dep_str}</span>
<span class="wl-detail">{w.required_cpu}cpu/{w.required_ram}GB | {w.current_latency_ms:.0f}ms / {w.sla_latency_ms:.0f}ms SLA | on: {w.assigned_server or 'NONE'}</span>
</div>'''

    traffic = "🟢 Active (seed-based)" if obs.traffic_multiplier_active else "⚪ Static"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Cloud Fleet Dashboard</title>
<meta http-equiv="refresh" content="2"><style>{_DASHBOARD_CSS}</style></head><body>
<h1>☁️ Cloud Fleet Dashboard</h1>
<div class="sub">Session: {sid} | Task: {env.task_id} | Step: {obs.step_number}/{obs.max_steps} | Traffic: {traffic}{' | 🔴 DONE' if obs.done else ''}</div>
<div class="metrics">
<div class="metric"><span class="label">Cost/hr</span><span class="value {cost_class}">${obs.total_cost_per_hour:.2f}</span></div>
<div class="metric"><span class="label">Budget/hr</span><span class="value">${obs.budget_per_hour:.2f}</span></div>
<div class="metric"><span class="label">SLA Violations</span><span class="value {sla_class}">{obs.sla_violations}</span></div>
<div class="metric"><span class="label">Unassigned</span><span class="value {un_class}">{obs.unassigned_workloads}</span></div>
<div class="metric"><span class="label">Score</span><span class="value">—</span></div>
</div>
<div class="grid">{server_cards}</div>
<div class="wl-section"><h2>Workloads</h2>{wl_items}</div>
<div style="text-align:center;margin-top:24px;color:#555;font-size:0.7rem">Auto-refreshes every 2s | {obs.message}</div>
</body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_endpoint(session_id: Optional[str] = Query(None)):
    env = sessions.get(session_id)
    if env is None:
        return HTMLResponse(f"""<!DOCTYPE html><html><head><style>{_DASHBOARD_CSS}</style></head><body>
<h1>☁️ Cloud Fleet Dashboard</h1><div class="sub">No active session. POST to /reset first, then visit /dashboard?session_id=YOUR_ID</div></body></html>""")
    sid = session_id or sessions.latest_id() or "?"
    with env._lock:
        return HTMLResponse(_render_dashboard(env, sid))


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
