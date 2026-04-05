"""
models.py — Pydantic v2 models for DevOps/FinOps OpenEnv environment.
Defines the Observation, Action, and Reward contracts.
These will be imported by env.py when we build the FastAPI server.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ── Instance Type Catalog ───────────────────────────────────────────
# nano:   1 CPU,  1 GB RAM, $0.05/hr
# micro:  1 CPU,  2 GB RAM, $0.10/hr
# small:  2 CPU,  4 GB RAM, $0.20/hr
# medium: 4 CPU,  8 GB RAM, $0.40/hr
# large:  8 CPU, 16 GB RAM, $0.80/hr
# xlarge: 16 CPU, 32 GB RAM, $1.60/hr

INSTANCE_CATALOG = {
    "nano":   {"cpu": 1,  "ram": 1,  "cost": 0.05},
    "micro":  {"cpu": 1,  "ram": 2,  "cost": 0.10},
    "small":  {"cpu": 2,  "ram": 4,  "cost": 0.20},
    "medium": {"cpu": 4,  "ram": 8,  "cost": 0.40},
    "large":  {"cpu": 8,  "ram": 16, "cost": 0.80},
    "xlarge": {"cpu": 16, "ram": 32, "cost": 1.60},
}

INSTANCE_TYPES = Literal["nano", "micro", "small", "medium", "large", "xlarge"]


# ── Sub-models ──────────────────────────────────────────────────────

class ServerInfo(BaseModel):
    """Snapshot of a single cloud server instance."""
    model_config = {"frozen": False}

    server_id: str = Field(
        description="Unique server identifier, e.g. 'srv-001'"
    )
    instance_type: str = Field(
        description="Instance tier: nano/micro/small/medium/large/xlarge"
    )
    cpu_cores: int = Field(
        description="Number of vCPU cores on this instance"
    )
    ram_gb: int = Field(
        description="RAM in GB on this instance"
    )
    cpu_utilization: float = Field(
        ge=0.0, le=1.0,
        description="Current CPU usage as a fraction, 0.0 (idle) to 1.0 (maxed)"
    )
    ram_utilization: float = Field(
        ge=0.0, le=1.0,
        description="Current RAM usage as a fraction, 0.0 (idle) to 1.0 (maxed)"
    )
    cost_per_hour: float = Field(
        ge=0.0,
        description="Hourly cost in USD for this instance"
    )
    assigned_workloads: List[str] = Field(
        default_factory=list,
        description="List of workload IDs currently running on this server"
    )


class WorkloadInfo(BaseModel):
    """Snapshot of a single workload/service that must be kept running."""
    model_config = {"frozen": False}

    workload_id: str = Field(
        description="Unique workload identifier, e.g. 'wl-api-gateway'"
    )
    name: str = Field(
        description="Human-readable workload name, e.g. 'API Gateway'"
    )
    required_cpu: float = Field(
        gt=0.0,
        description="Minimum CPU cores this workload needs to run"
    )
    required_ram: float = Field(
        gt=0.0,
        description="Minimum RAM in GB this workload needs to run"
    )
    current_latency_ms: float = Field(
        ge=0.0,
        description="Current p95 response latency in milliseconds"
    )
    sla_latency_ms: float = Field(
        gt=0.0,
        description="Maximum acceptable p95 latency before SLA breach"
    )
    is_critical: bool = Field(
        description="If True, SLA breach on this workload carries 2x penalty weight"
    )
    assigned_server: Optional[str] = Field(
        default=None,
        description="Server ID where this workload runs, or null if unassigned (DOWN)"
    )


# ── Core Models ─────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    Everything the agent sees at each step.
    Provides full visibility into infrastructure state, cost, and performance.
    """
    model_config = {"frozen": False}

    servers: List[ServerInfo] = Field(
        description="All active server instances in the fleet"
    )
    workloads: List[WorkloadInfo] = Field(
        description="All workloads that must be kept running"
    )
    total_cost_per_hour: float = Field(
        ge=0.0,
        description="Sum of all running server costs in USD/hr"
    )
    budget_per_hour: float = Field(
        gt=0.0,
        description="Target budget the agent should optimize toward in USD/hr"
    )
    sla_violations: int = Field(
        ge=0,
        description="Count of workloads currently breaching their latency SLA"
    )
    unassigned_workloads: int = Field(
        ge=0,
        description="Count of workloads not running on any server (effectively DOWN)"
    )
    step_number: int = Field(
        ge=0,
        description="Current step in the episode (0-indexed)"
    )
    max_steps: int = Field(
        gt=0,
        description="Maximum steps allowed before episode ends"
    )
    done: bool = Field(
        description="True if the episode is over"
    )
    message: str = Field(
        description="Human-readable feedback about the last action's effect"
    )


class Action(BaseModel):
    """
    A single infrastructure action the agent takes each step.

    Valid combinations:
      provision  -> instance_type required
      terminate  -> server_id required
      resize     -> server_id + instance_type required
      migrate    -> workload_id + target_server_id required
      noop       -> no additional fields needed
    """
    model_config = {"frozen": False}

    action_type: Literal["provision", "terminate", "resize", "migrate", "noop"] = Field(
        description="The infrastructure operation to perform"
    )
    server_id: Optional[str] = Field(
        default=None,
        description="Target server for terminate/resize actions"
    )
    instance_type: Optional[INSTANCE_TYPES] = Field(
        default=None,
        description="Instance tier for provision/resize actions"
    )
    workload_id: Optional[str] = Field(
        default=None,
        description="Workload to move for migrate actions"
    )
    target_server_id: Optional[str] = Field(
        default=None,
        description="Destination server for migrate actions"
    )


class Reward(BaseModel):
    """
    Scoring signal returned at every step.
    Carries partial credit — the agent learns from incremental progress.

    Formula: score = (cost_efficiency * 0.50) + (performance_score * 0.40) - (penalty * 0.10)
    """
    model_config = {"frozen": False}

    score: float = Field(
        ge=0.0, le=1.0,
        description="Overall step score combining cost efficiency, performance, and penalties"
    )
    cost_efficiency: float = Field(
        ge=0.0, le=1.0,
        description="How close current cost is to budget target. "
                    "1.0 = at or under budget, scales down as overspend increases"
    )
    performance_score: float = Field(
        ge=0.0, le=1.0,
        description="Fraction of workloads meeting SLA with no downtime. "
                    "1.0 = all workloads healthy and within latency SLA"
    )
    penalty: float = Field(
        ge=0.0, le=1.0,
        description="Deductions for destructive/wasteful actions. "
                    "0.0 = no penalty, 1.0 = maximum penalty"
    )
    is_final: bool = Field(
        description="True if this is the terminal reward for the episode"
    )
    breakdown: str = Field(
        description="Human-readable explanation of how this score was calculated"
    )
