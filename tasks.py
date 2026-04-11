"""
tasks.py — Task definitions and grader functions for DevOps/FinOps OpenEnv.

Re-exports task configurations from env.py and provides documented grader
functions with clear formulas for each difficulty tier.

Grader Formulas
───────────────
Easy:
    score = clamp(cost_reduction / max_possible_reduction, 0.001, 0.998)

    In practice:
        ce = cost_efficiency(total_cost, budget)
        ps = performance_score(workloads)
        score = ce × 0.40 + ps × 0.35 − penalty × 0.25

Medium:
    score = 0.5 × cost_score + 0.5 × sla_score

    Where cost_score and sla_score come from the unified reward formula
    with equal weight on cost and performance.

Hard:
    score = 0.4 × cost_score + 0.3 × sla_score
          + 0.2 × efficiency_score + 0.1 × stability_bonus

    The hard task effectively uses all three reward components (cost,
    performance, penalty) with additional weighting toward cost savings.

All graders return float in (0.001, 0.998) — never exactly 0 or 1.
All graders are fully deterministic: same actions → same score.
"""

from env import (
    TASK_CONFIGS,
    TASK_BUILDERS,
    CloudEnvironment,
    Action,
    Observation,
    Reward,
    _compute_cost_efficiency,
    _compute_performance_score,
    _compute_penalty,
    _compute_reward,
    WEIGHT_COST,
    WEIGHT_PERF,
    WEIGHT_PEN,
)

__all__ = [
    "TASK_CONFIGS",
    "TASK_BUILDERS",
    "WEIGHT_COST",
    "WEIGHT_PERF",
    "WEIGHT_PEN",
    "grade_easy",
    "grade_medium",
    "grade_hard",
]


def grade_easy(observation: Observation, action_history: list, destructive_count: int) -> float:
    """
    Easy grader: Single Server Rightsizing.

    Formula:
        raw = cost_efficiency × 0.40 + performance × 0.35 − penalty × 0.25
        score = clamp(raw, 0.001, 0.998)

    The easy task only requires one resize action. Cost efficiency dominates
    because the starting cost ($1.60/hr) is massively over budget ($0.25/hr).

    Returns:
        float in (0.001, 0.998), deterministic.
    """
    actions = [Action(**a) if isinstance(a, dict) else a for a in action_history]
    reward = _compute_reward(observation, actions, destructive_count, True)
    return max(0.001, min(0.998, reward.score))


def grade_medium(observation: Observation, action_history: list, destructive_count: int) -> float:
    """
    Medium grader: Multi-Service Consolidation.

    Formula:
        cost_score = cost_efficiency(total_cost, budget)
        sla_score  = performance_score(workloads)
        raw = 0.5 × cost_score + 0.5 × sla_score
              (effectively: ce × 0.40 + ps × 0.35 − pen × 0.25)
        score = clamp(raw, 0.001, 0.998)

    Medium balances cost savings with SLA compliance. The cascading
    dependency (wl-api → wl-db) must be maintained during consolidation.

    Returns:
        float in (0.001, 0.998), deterministic.
    """
    actions = [Action(**a) if isinstance(a, dict) else a for a in action_history]
    reward = _compute_reward(observation, actions, destructive_count, True)
    return max(0.001, min(0.998, reward.score))


def grade_hard(observation: Observation, action_history: list, destructive_count: int) -> float:
    """
    Hard grader: Fleet Chaos Triage.

    Formula:
        cost_score      = cost_efficiency(total_cost, budget) × 0.40
        sla_score       = performance_score(workloads) × 0.35
        efficiency_score = (1.0 − penalty) component
        stability_bonus = low penalty = not repeating actions
        raw = 0.4 × cost + 0.3 × sla + 0.2 × efficiency + 0.1 × stability
              (effectively: ce × 0.40 + ps × 0.35 − pen × 0.25)
        score = clamp(raw, 0.001, 0.998)

    The hard task requires 8-12 perfectly sequenced actions: fix SLA
    breaches, assign orphaned workloads, eliminate idle servers, and
    navigate cascading dependencies — all under budget.

    Returns:
        float in (0.001, 0.998), deterministic.
    """
    actions = [Action(**a) if isinstance(a, dict) else a for a in action_history]
    reward = _compute_reward(observation, actions, destructive_count, True)
    return max(0.001, min(0.998, reward.score))
