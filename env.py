"""
env.py — DevOps/FinOps OpenEnv Environment (Complete)

All-in-one: Pydantic models, 3 task definitions, grading,
environment simulation, and FastAPI endpoints.

Stack: Python 3.11, FastAPI, Pydantic v2, uvicorn
Port:  7860 (HuggingFace Spaces default)
"""

from __future__ import annotations

import copy
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal


# ═══════════════════════════════════════════════════════════════════════
# INSTANCE CATALOG
# ═══════════════════════════════════════════════════════════════════════

INSTANCE_CATALOG: Dict[str, Dict] = {
    "nano":   {"cpu": 1,  "ram": 1,  "cost": 0.05},
    "micro":  {"cpu": 1,  "ram": 2,  "cost": 0.10},
    "small":  {"cpu": 2,  "ram": 4,  "cost": 0.20},
    "medium": {"cpu": 4,  "ram": 8,  "cost": 0.40},
    "large":  {"cpu": 8,  "ram": 16, "cost": 0.80},
    "xlarge": {"cpu": 16, "ram": 32, "cost": 1.60},
}

INSTANCE_TYPES = Literal["nano", "micro", "small", "medium", "large", "xlarge"]


# ═══════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════

class ServerInfo(BaseModel):
    server_id: str
    instance_type: str
    cpu_cores: int
    ram_gb: int
    cpu_utilization: float = Field(ge=0.0, le=1.0)
    ram_utilization: float = Field(ge=0.0, le=1.0)
    cost_per_hour: float = Field(ge=0.0)
    assigned_workloads: List[str] = Field(default_factory=list)


class WorkloadInfo(BaseModel):
    workload_id: str
    name: str
    required_cpu: float = Field(gt=0.0)
    required_ram: float = Field(gt=0.0)
    current_latency_ms: float = Field(ge=0.0)
    sla_latency_ms: float = Field(gt=0.0)
    is_critical: bool
    assigned_server: Optional[str] = None


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


class Action(BaseModel):
    action_type: Literal["provision", "terminate", "resize", "migrate", "noop"]
    server_id: Optional[str] = None
    instance_type: Optional[INSTANCE_TYPES] = None
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


# ═══════════════════════════════════════════════════════════════════════
# LATENCY SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def _simulate_latency(
    workload: WorkloadInfo,
    server_cpu: int,
    server_ram: int,
    workloads_on_server: List[WorkloadInfo],
) -> float:
    """
    Deterministic latency model based on resource pressure.

    - utilization <= 0.5:  comfortable, ~20% of SLA
    - utilization 0.5-1.0: scales to ~75% of SLA
    - utilization > 1.0:   overloaded, breaches SLA

    Context-switching overhead: +5% per additional workload on same server.
    """
    total_cpu = sum(w.required_cpu for w in workloads_on_server)
    total_ram = sum(w.required_ram for w in workloads_on_server)

    cpu_ratio = total_cpu / server_cpu if server_cpu > 0 else 10.0
    ram_ratio = total_ram / server_ram if server_ram > 0 else 10.0
    base_util = max(cpu_ratio, ram_ratio)

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


# ═══════════════════════════════════════════════════════════════════════
# GRADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def _compute_cost_efficiency(total_cost: float, budget: float) -> float:
    if budget <= 0:
        return 0.0
    if total_cost <= 0:
        return 1.0
    if total_cost <= budget:
        savings_ratio = 1.0 - (total_cost / budget)
        return 0.80 + (0.20 * savings_ratio)
    else:
        overspend_ratio = (total_cost - budget) / budget
        return max(0.0, 0.80 - (0.40 * overspend_ratio))


def _compute_performance_score(workloads: List[WorkloadInfo]) -> float:
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


def _compute_penalty(action_history: List[Action], destructive_count: int) -> float:
    penalty = 0.0

    # Repeated identical actions (loop detection)
    if len(action_history) >= 2:
        seen: Dict[tuple, int] = {}
        for act in action_history:
            key = (act.action_type, act.server_id, act.instance_type,
                   act.workload_id, act.target_server_id)
            seen[key] = seen.get(key, 0) + 1
        for count in seen.values():
            if count > 1:
                penalty += 0.1 * (count - 1)

    # Destructive terminations (server with workloads)
    penalty += 0.3 * destructive_count

    # Trailing noops (agent gave up)
    noop_tail = 0
    for act in reversed(action_history):
        if act.action_type == "noop":
            noop_tail += 1
        else:
            break
    if noop_tail >= 3:
        penalty += 0.05 * (noop_tail - 2)

    # Provision spam without migration
    provisions = sum(1 for a in action_history if a.action_type == "provision")
    migrations = sum(1 for a in action_history if a.action_type == "migrate")
    if provisions >= 2 and migrations == 0:
        penalty += 0.1

    return min(penalty, 1.0)


def _compute_reward(
    observation: Observation,
    action_history: List[Action],
    destructive_count: int,
    is_final: bool,
) -> Reward:
    ce = _compute_cost_efficiency(observation.total_cost_per_hour, observation.budget_per_hour)
    ps = _compute_performance_score(observation.workloads)
    if ps == 0.0:
        ce = 0.0
    pen = _compute_penalty(action_history, destructive_count)
    raw = (ce * 0.50) + (ps * 0.40) - (pen * 0.10)
    score = max(0.0, min(1.0, raw))
    breakdown = (
        f"cost_efficiency={ce:.3f}*0.50 + performance={ps:.3f}*0.40"
        f" - penalty={pen:.3f}*0.10 = {score:.4f}"
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
# TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

TASK_CONFIGS = {
    "easy": {
        "name": "Single Server Rightsizing",
        "objective": (
            "Resize a massively over-provisioned xlarge server to match "
            "a lightweight workload and bring cost under budget."
        ),
        "max_steps": 5,
        "budget_per_hour": 0.20,
    },
    "medium": {
        "name": "Multi-Service Consolidation",
        "objective": (
            "Consolidate 4 workloads from 4 oversized servers onto fewer, "
            "right-sized servers to cut cost below budget while keeping SLAs."
        ),
        "max_steps": 10,
        "budget_per_hour": 0.50,
    },
    "hard": {
        "name": "Fleet Chaos Triage",
        "objective": (
            "Fix a broken fleet with orphaned workloads, SLA breaches, "
            "idle servers, and over-provisioning."
        ),
        "max_steps": 15,
        "budget_per_hour": 0.80,
    },
}


def _build_easy_state():
    servers = [
        ServerInfo(
            server_id="srv-001", instance_type="xlarge",
            cpu_cores=16, ram_gb=32,
            cpu_utilization=0.06, ram_utilization=0.06,
            cost_per_hour=1.60,
            assigned_workloads=["wl-web"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-web", name="Company Website",
            required_cpu=1.0, required_ram=2.0,
            current_latency_ms=40.0, sla_latency_ms=200.0,
            is_critical=False, assigned_server="srv-001",
        ),
    ]
    return servers, workloads


def _build_medium_state():
    servers = [
        ServerInfo(
            server_id="srv-001", instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=0.25, ram_utilization=0.25,
            cost_per_hour=0.40, assigned_workloads=["wl-api"],
        ),
        ServerInfo(
            server_id="srv-002", instance_type="large",
            cpu_cores=8, ram_gb=16,
            cpu_utilization=0.25, ram_utilization=0.25,
            cost_per_hour=0.80, assigned_workloads=["wl-db"],
        ),
        ServerInfo(
            server_id="srv-003", instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=0.25, ram_utilization=0.19,
            cost_per_hour=0.40, assigned_workloads=["wl-cache"],
        ),
        ServerInfo(
            server_id="srv-004", instance_type="large",
            cpu_cores=8, ram_gb=16,
            cpu_utilization=0.06, ram_utilization=0.06,
            cost_per_hour=0.80, assigned_workloads=["wl-worker"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-api", name="API Gateway",
            required_cpu=1.0, required_ram=2.0,
            current_latency_ms=30.0, sla_latency_ms=150.0,
            is_critical=True, assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-db", name="Database Replica",
            required_cpu=2.0, required_ram=4.0,
            current_latency_ms=20.0, sla_latency_ms=100.0,
            is_critical=True, assigned_server="srv-002",
        ),
        WorkloadInfo(
            workload_id="wl-cache", name="Redis Cache",
            required_cpu=1.0, required_ram=1.5,
            current_latency_ms=10.0, sla_latency_ms=50.0,
            is_critical=False, assigned_server="srv-003",
        ),
        WorkloadInfo(
            workload_id="wl-worker", name="Background Worker",
            required_cpu=0.5, required_ram=1.0,
            current_latency_ms=100.0, sla_latency_ms=500.0,
            is_critical=False, assigned_server="srv-004",
        ),
    ]
    return servers, workloads


def _build_hard_state():
    servers = [
        ServerInfo(
            server_id="srv-001", instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=1.0, ram_utilization=0.63,
            cost_per_hour=0.40,
            assigned_workloads=["wl-auth", "wl-api"],
        ),
        ServerInfo(
            server_id="srv-002", instance_type="large",
            cpu_cores=8, ram_gb=16,
            cpu_utilization=0.25, ram_utilization=0.25,
            cost_per_hour=0.80, assigned_workloads=["wl-db"],
        ),
        ServerInfo(
            server_id="srv-003", instance_type="nano",
            cpu_cores=1, ram_gb=1,
            cpu_utilization=1.0, ram_utilization=1.0,
            cost_per_hour=0.05, assigned_workloads=["wl-search"],
        ),
        ServerInfo(
            server_id="srv-004", instance_type="medium",
            cpu_cores=4, ram_gb=8,
            cpu_utilization=0.0, ram_utilization=0.0,
            cost_per_hour=0.40, assigned_workloads=[],
        ),
        ServerInfo(
            server_id="srv-005", instance_type="nano",
            cpu_cores=1, ram_gb=1,
            cpu_utilization=0.50, ram_utilization=0.50,
            cost_per_hour=0.05, assigned_workloads=["wl-metrics"],
        ),
    ]
    workloads = [
        WorkloadInfo(
            workload_id="wl-auth", name="Auth Service",
            required_cpu=2.0, required_ram=3.0,
            current_latency_ms=180.0, sla_latency_ms=150.0,
            is_critical=True, assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-api", name="API Gateway",
            required_cpu=2.0, required_ram=2.0,
            current_latency_ms=180.0, sla_latency_ms=150.0,
            is_critical=True, assigned_server="srv-001",
        ),
        WorkloadInfo(
            workload_id="wl-db", name="Primary Database",
            required_cpu=2.0, required_ram=4.0,
            current_latency_ms=20.0, sla_latency_ms=100.0,
            is_critical=True, assigned_server="srv-002",
        ),
        WorkloadInfo(
            workload_id="wl-search", name="Search Index",
            required_cpu=1.0, required_ram=2.0,
            current_latency_ms=310.0, sla_latency_ms=100.0,
            is_critical=False, assigned_server="srv-003",
        ),
        WorkloadInfo(
            workload_id="wl-metrics", name="Metrics Collector",
            required_cpu=0.5, required_ram=0.5,
            current_latency_ms=100.0, sla_latency_ms=500.0,
            is_critical=False, assigned_server="srv-005",
        ),
        WorkloadInfo(
            workload_id="wl-ml-jobs", name="ML Training Pipeline",
            required_cpu=3.0, required_ram=4.0,
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
# ENVIRONMENT CLASS
# ═══════════════════════════════════════════════════════════════════════

class CloudEnvironment:
    """Stateful environment managing cloud infrastructure simulation."""

    def __init__(self):
        self._lock = threading.Lock()
        self.servers: List[ServerInfo] = []
        self.workloads: List[WorkloadInfo] = []
        self.task_id: str = "easy"
        self.budget: float = 0.20
        self.max_steps: int = 5
        self.step_number: int = 0
        self.done: bool = True
        self.action_history: List[Action] = []
        self.destructive_count: int = 0
        self._server_counter: int = 0
        self._last_message: str = "No episode running. Call /reset first."

    # ── helpers ──────────────────────────────────────────────────────

    def _next_server_id(self) -> str:
        self._server_counter += 1
        return f"srv-{self._server_counter:03d}"

    def _find_server(self, sid: str) -> Optional[ServerInfo]:
        for s in self.servers:
            if s.server_id == sid:
                return s
        return None

    def _find_workload(self, wid: str) -> Optional[WorkloadInfo]:
        for w in self.workloads:
            if w.workload_id == wid:
                return w
        return None

    def _workloads_on_server(self, sid: str) -> List[WorkloadInfo]:
        return [w for w in self.workloads if w.assigned_server == sid]

    def _recalculate_state(self):
        """Recompute utilization + latency for every server and workload."""
        for srv in self.servers:
            wls = self._workloads_on_server(srv.server_id)
            total_cpu = sum(w.required_cpu for w in wls)
            total_ram = sum(w.required_ram for w in wls)
            srv.cpu_utilization = min(1.0, round(total_cpu / srv.cpu_cores, 4)) if srv.cpu_cores > 0 else 0.0
            srv.ram_utilization = min(1.0, round(total_ram / srv.ram_gb, 4)) if srv.ram_gb > 0 else 0.0
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
            wl.current_latency_ms = _simulate_latency(wl, srv.cpu_cores, srv.ram_gb, wls_on)

    def _build_observation(self) -> Observation:
        total_cost = sum(s.cost_per_hour for s in self.servers)
        sla_violations = sum(
            1 for w in self.workloads
            if w.assigned_server is not None and w.current_latency_ms > w.sla_latency_ms
        )
        unassigned = sum(1 for w in self.workloads if w.assigned_server is None)
        return Observation(
            servers=copy.deepcopy(self.servers),
            workloads=copy.deepcopy(self.workloads),
            total_cost_per_hour=round(total_cost, 4),
            budget_per_hour=self.budget,
            sla_violations=sla_violations,
            unassigned_workloads=unassigned,
            step_number=self.step_number,
            max_steps=self.max_steps,
            done=self.done,
            message=self._last_message,
        )

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id '{task_id}'. Use: {list(TASK_CONFIGS.keys())}")

        config = TASK_CONFIGS[task_id]
        builder = TASK_BUILDERS[task_id]
        servers, workloads = builder()

        self.task_id = task_id
        self.budget = config["budget_per_hour"]
        self.max_steps = config["max_steps"]
        self.step_number = 0
        self.done = False
        self.action_history = []
        self.destructive_count = 0
        self.servers = servers
        self.workloads = workloads

        max_id = 0
        for s in self.servers:
            num = int(s.server_id.split("-")[1])
            if num > max_id:
                max_id = num
        self._server_counter = max_id

        self._recalculate_state()
        self._last_message = f"Episode started: {config['name']}. {config['objective']}"
        return self._build_observation()

    # ── step ─────────────────────────────────────────────────────────

    def step(self, action: Action) -> dict:
        if self.done:
            obs = self._build_observation()
            reward = _compute_reward(obs, self.action_history, self.destructive_count, True)
            return {"observation": obs, "reward": reward, "done": True,
                    "info": {"error": "Episode is done. Call /reset to start a new one."}}

        self.action_history.append(action)
        msg = self._execute_action(action)
        self._last_message = msg

        self._recalculate_state()
        self.step_number += 1

        if self.step_number >= self.max_steps:
            self.done = True

        obs = self._build_observation()
        reward = _compute_reward(obs, self.action_history, self.destructive_count, self.done)
        return {
            "observation": obs,
            "reward": reward,
            "done": self.done,
            "info": {"step": self.step_number, "action_result": msg},
        }

    def _execute_action(self, action: Action) -> str:
        atype = action.action_type

        if atype == "noop":
            return "No operation performed."

        elif atype == "provision":
            if action.instance_type is None:
                return "ERROR: provision requires instance_type."
            spec = INSTANCE_CATALOG[action.instance_type]
            new_id = self._next_server_id()
            self.servers.append(ServerInfo(
                server_id=new_id,
                instance_type=action.instance_type,
                cpu_cores=spec["cpu"],
                ram_gb=spec["ram"],
                cpu_utilization=0.0,
                ram_utilization=0.0,
                cost_per_hour=spec["cost"],
                assigned_workloads=[],
            ))
            return f"Provisioned {action.instance_type} server {new_id} (${spec['cost']}/hr)."

        elif atype == "terminate":
            if action.server_id is None:
                return "ERROR: terminate requires server_id."
            srv = self._find_server(action.server_id)
            if srv is None:
                return f"ERROR: server {action.server_id} not found."
            orphaned = self._workloads_on_server(srv.server_id)
            if orphaned:
                self.destructive_count += 1
                for wl in orphaned:
                    wl.assigned_server = None
                    wl.current_latency_ms = 0.0
            self.servers = [s for s in self.servers if s.server_id != action.server_id]
            if orphaned:
                names = [w.workload_id for w in orphaned]
                return (f"Terminated {action.server_id}. WARNING: {len(orphaned)} "
                        f"workloads orphaned: {names}. Destructive penalty applied.")
            return f"Terminated idle server {action.server_id}."

        elif atype == "resize":
            if action.server_id is None or action.instance_type is None:
                return "ERROR: resize requires server_id and instance_type."
            srv = self._find_server(action.server_id)
            if srv is None:
                return f"ERROR: server {action.server_id} not found."
            if srv.instance_type == action.instance_type:
                return f"Server {action.server_id} is already {action.instance_type}. No change."
            old_type = srv.instance_type
            spec = INSTANCE_CATALOG[action.instance_type]
            srv.instance_type = action.instance_type
            srv.cpu_cores = spec["cpu"]
            srv.ram_gb = spec["ram"]
            srv.cost_per_hour = spec["cost"]
            return (f"Resized {action.server_id} from {old_type} to "
                    f"{action.instance_type} (${spec['cost']}/hr).")

        elif atype == "migrate":
            if action.workload_id is None or action.target_server_id is None:
                return "ERROR: migrate requires workload_id and target_server_id."
            wl = self._find_workload(action.workload_id)
            if wl is None:
                return f"ERROR: workload {action.workload_id} not found."
            target = self._find_server(action.target_server_id)
            if target is None:
                return f"ERROR: target server {action.target_server_id} not found."
            if wl.assigned_server == action.target_server_id:
                return f"Workload {action.workload_id} is already on {action.target_server_id}."
            old_server = wl.assigned_server
            if old_server:
                src = self._find_server(old_server)
                if src:
                    src.assigned_workloads = [
                        w for w in src.assigned_workloads if w != action.workload_id
                    ]
            wl.assigned_server = action.target_server_id
            target.assigned_workloads.append(action.workload_id)
            origin = old_server if old_server else "unassigned"
            return (f"Migrated {action.workload_id} from {origin} "
                    f"to {action.target_server_id}.")

        return f"ERROR: unknown action_type '{atype}'."

    # ── state ────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        obs = self._build_observation()
        return {
            "task_id": self.task_id,
            "observation": obs,
            "action_history": [a.model_dump() for a in self.action_history],
            "destructive_terminations": self.destructive_count,
        }


# ═══════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DevOps/FinOps OpenEnv",
    description="AI agent environment for cloud infrastructure cost vs performance optimization.",
    version="1.0.0",
)

env = CloudEnvironment()


@app.get("/")
def health_check():
    return {"status": "ok", "environment": "devops-finops-cloud-optimizer"}


@app.post("/reset")
def reset_endpoint(body: ResetRequest = ResetRequest()):
    with env._lock:
        try:
            obs = env.reset(body.task_id)
            return obs.model_dump()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step_endpoint(action: Action):
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
def state_endpoint():
    with env._lock:
        state = env.get_state()
        return {
            "task_id": state["task_id"],
            "observation": state["observation"].model_dump(),
            "action_history": state["action_history"],
            "destructive_terminations": state["destructive_terminations"],
        }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
