"""
test_env.py — Unit tests for DevOps/FinOps OpenEnv environment.

Validates all 3 fixed tasks across multiple scenarios to ensure:
- Graders return float 0.0–1.0
- Different actions produce different scores (no flat grading)
- Deterministic: same actions → same score
- Penalty system catches degenerate strategies
- Session management works
- Spot eviction and cascading failures behave correctly

Run: python test_env.py
"""

import copy
import json
import sys

from env import (
    CloudEnvironment, Action, SessionManager,
    _compute_cost_efficiency, _compute_performance_score, _compute_penalty,
    _simulate_latency, _apply_cascading_failures, _get_traffic_multiplier,
    INSTANCE_CATALOG, TASK_CONFIGS, WorkloadInfo,
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def test_instance_catalog():
    print("\n═══ Instance Catalog ═══")
    check("Has 11 instance types", len(INSTANCE_CATALOG) == 11, f"Got {len(INSTANCE_CATALOG)}")
    for name, spec in INSTANCE_CATALOG.items():
        check(f"{name} has all fields",
              all(k in spec for k in ["cpu", "ram", "disk_iops", "net_gbps", "cost", "is_spot"]),
              f"Missing fields in {name}")
    spot_types = [k for k, v in INSTANCE_CATALOG.items() if v["is_spot"]]
    check("Has spot instances", len(spot_types) >= 2, f"Spot types: {spot_types}")
    for st in spot_types:
        non_spot_equiv = st.replace("spot-", "")
        if non_spot_equiv in INSTANCE_CATALOG:
            check(f"{st} cheaper than {non_spot_equiv}",
                  INSTANCE_CATALOG[st]["cost"] < INSTANCE_CATALOG[non_spot_equiv]["cost"])


def test_cost_efficiency():
    print("\n═══ Cost Efficiency ═══")
    budget = 0.50
    check("At zero cost → 1.0", _compute_cost_efficiency(0.0, budget) == 1.0)
    check("At budget → 0.90", _compute_cost_efficiency(budget, budget) == 0.90)
    check("Under budget > at budget",
          _compute_cost_efficiency(0.25, budget) > _compute_cost_efficiency(budget, budget))
    check("Over budget < at budget",
          _compute_cost_efficiency(0.80, budget) < _compute_cost_efficiency(budget, budget))
    check("At 2x budget → 0.0", _compute_cost_efficiency(1.00, budget) == 0.0)
    check("Monotonically decreasing",
          _compute_cost_efficiency(0.10, budget) >
          _compute_cost_efficiency(0.30, budget) >
          _compute_cost_efficiency(0.50, budget) >
          _compute_cost_efficiency(0.80, budget))


def test_performance_score():
    print("\n═══ Performance Score ═══")
    healthy = WorkloadInfo(
        workload_id="w1", name="t", required_cpu=1, required_ram=1,
        current_latency_ms=10, sla_latency_ms=100, is_critical=False, assigned_server="s1")
    breach = WorkloadInfo(
        workload_id="w2", name="t", required_cpu=1, required_ram=1,
        current_latency_ms=200, sla_latency_ms=100, is_critical=False, assigned_server="s1")
    down = WorkloadInfo(
        workload_id="w3", name="t", required_cpu=1, required_ram=1,
        current_latency_ms=0, sla_latency_ms=100, is_critical=False, assigned_server=None)
    crit_breach = WorkloadInfo(
        workload_id="w4", name="t", required_cpu=1, required_ram=1,
        current_latency_ms=200, sla_latency_ms=100, is_critical=True, assigned_server="s1")

    check("All healthy → 1.0", _compute_performance_score([healthy]) == 1.0)
    check("All breach → 0.0", _compute_performance_score([breach]) == 0.0)
    check("All down → 0.0", _compute_performance_score([down]) == 0.0)
    check("Critical breach weighted 2x",
          _compute_performance_score([healthy, crit_breach]) < _compute_performance_score([healthy, breach]))


def test_penalty():
    print("\n═══ Penalty System ═══")
    noop = Action(action_type="noop")
    resize = Action(action_type="resize", server_id="s1", instance_type="micro")

    check("No actions → 0 penalty", _compute_penalty([], 0) == 0.0)
    check("Single action → 0 penalty", _compute_penalty([resize], 0) == 0.0)
    check("Repeated action → penalty > 0", _compute_penalty([resize, resize], 0) > 0.0)
    check("Destructive termination → 0.50 penalty", _compute_penalty([], 1) == 0.50)
    check("2 destructive → 1.0 (clamped)", _compute_penalty([], 2) == 1.0)
    check("3 trailing noops after useful action → NO penalty", _compute_penalty([resize, noop, noop, noop], 0) == 0.0)
    check("All-noop idleness (3 noops, no useful actions) → penalty",
          _compute_penalty([noop, noop, noop], 0) > 0.0)
    pen_3noop_idle = _compute_penalty([noop, noop, noop], 0)
    pen_5noop_idle = _compute_penalty([noop, noop, noop, noop, noop], 0)
    check("More idle noops → more penalty", pen_5noop_idle > pen_3noop_idle)
    prov = Action(action_type="provision", instance_type="medium")
    check("2 provisions no migrate → penalty", _compute_penalty([prov, prov], 0) > 0.0)


def test_easy_task():
    print("\n═══ Easy Task — Single Server Rightsizing ═══")
    env = CloudEnvironment()

    # Scenario 1: no-action (all noops)
    obs = env.reset("easy")
    check("Easy starts with 2 workloads", len(obs.workloads) == 2)
    check("Easy starts with 1 server", len(obs.servers) == 1)
    check("Easy budget = $0.25", obs.budget_per_hour == 0.25)
    for _ in range(5):
        r = env.step(Action(action_type="noop"))
    s_noop = r["reward"].score
    check(f"Easy no-action score={s_noop:.4f} in (0, 1)", 0.0 < s_noop < 1.0)

    # Scenario 2: resize to small (optimal)
    env.reset("easy")
    r = env.step(Action(action_type="resize", server_id="srv-001", instance_type="small"))
    s_small = r["reward"].score
    check(f"Easy resize→small score={s_small:.4f} > noop score={s_noop:.4f}", s_small > s_noop)

    # Scenario 3: resize to micro (too small — breaches SLA)
    env.reset("easy")
    r = env.step(Action(action_type="resize", server_id="srv-001", instance_type="micro"))
    s_micro = r["reward"].score
    check(f"Easy resize→micro score={s_micro:.4f} != resize→small", abs(s_micro - s_small) > 0.01)

    # Scenario 4: destructive terminate
    env.reset("easy")
    r = env.step(Action(action_type="terminate", server_id="srv-001"))
    s_destroy = r["reward"].score
    check(f"Easy destructive terminate score={s_destroy:.4f} < noop", s_destroy < s_noop)


def test_medium_task():
    print("\n═══ Medium Task — Multi-Service Consolidation ═══")
    env = CloudEnvironment()

    # Scenario 1: no-action
    obs = env.reset("medium")
    check("Medium has 4 workloads", len(obs.workloads) == 4)
    check("Medium has 4 servers", len(obs.servers) == 4)
    check("Medium has dependency", any(w.dependencies for w in obs.workloads))
    for _ in range(10):
        r = env.step(Action(action_type="noop"))
    s_noop = r["reward"].score
    check(f"Medium no-action score={s_noop:.4f}", 0.0 < s_noop < 1.0)

    # Scenario 2: consolidate to 1 large
    env.reset("medium")
    for wl_id in ["wl-api", "wl-cache", "wl-worker"]:
        env.step(Action(action_type="migrate", workload_id=wl_id, target_server_id="srv-002"))
    for sid in ["srv-001", "srv-003", "srv-004"]:
        env.step(Action(action_type="terminate", server_id=sid))
    while env.step_number < 10:
        r = env.step(Action(action_type="noop"))
    s_consol = r["reward"].score
    check(f"Medium consolidate score={s_consol:.4f} > noop", s_consol > s_noop)


def test_hard_task():
    print("\n═══ Hard Task — Fleet Chaos Triage ═══")
    env = CloudEnvironment()

    obs = env.reset("hard")
    check("Hard has 6 workloads", len(obs.workloads) == 6)
    check("Hard has 5 servers", len(obs.servers) == 5)
    check("Hard starts with SLA violations", obs.sla_violations >= 3)
    check("Hard starts with unassigned workloads", obs.unassigned_workloads >= 1)
    check("Hard budget = $1.20", obs.budget_per_hour == 1.20)
    check("Hard has cascading dependencies",
          sum(1 for w in obs.workloads if w.dependencies) >= 2)

    # Scenario 1: no-action
    for _ in range(15):
        r = env.step(Action(action_type="noop"))
    s_noop = r["reward"].score
    check(f"Hard no-action score={s_noop:.4f}", 0.0 < s_noop < 1.0)

    # Scenario 2: partial fix — migrate wl-api off overloaded srv-001
    env.reset("hard")
    env.step(Action(action_type="migrate", workload_id="wl-api", target_server_id="srv-004"))
    env.step(Action(action_type="migrate", workload_id="wl-search", target_server_id="srv-004"))
    env.step(Action(action_type="migrate", workload_id="wl-ml-jobs", target_server_id="srv-002"))
    while env.step_number < 15:
        r = env.step(Action(action_type="noop"))
    s_partial = r["reward"].score
    check(f"Hard partial-fix score={s_partial:.4f} > noop={s_noop:.4f}", s_partial > s_noop)


def test_determinism():
    print("\n═══ Determinism ═══")
    env1 = CloudEnvironment()
    env2 = CloudEnvironment()

    env1.reset("hard")
    env2.reset("hard")
    r1 = env1.step(Action(action_type="migrate", workload_id="wl-api", target_server_id="srv-004"))
    r2 = env2.step(Action(action_type="migrate", workload_id="wl-api", target_server_id="srv-004"))
    check("Same actions → same score",
          r1["reward"].score == r2["reward"].score,
          f"{r1['reward'].score} != {r2['reward'].score}")
    check("Same actions → same cost",
          r1["observation"].total_cost_per_hour == r2["observation"].total_cost_per_hour)


def test_session_management():
    print("\n═══ Session Management ═══")
    sm = SessionManager()
    s1 = sm.create()
    s2 = sm.create()
    check("Creates unique session IDs", s1 != s2)
    check("Can retrieve session 1", sm.get(s1) is not None)
    check("Can retrieve session 2", sm.get(s2) is not None)
    check("Unknown session returns None", sm.get("nonexistent") is None)
    check("Latest is most recent", sm.latest_id() == s2)

    env1 = sm.get(s1)
    env2 = sm.get(s2)
    env1.reset("easy")
    env2.reset("hard")
    check("Sessions are independent", env1.task_id != env2.task_id)


def test_procedural_generation():
    print("\n═══ Procedural Generation ═══")
    env = CloudEnvironment()
    obs1 = env.reset("random", seed=42)
    check("Random task creates servers", len(obs1.servers) >= 3)
    check("Random task creates workloads", len(obs1.workloads) >= 4)
    check("Random task has budget", obs1.budget_per_hour > 0)

    obs2 = env.reset("random", seed=42)
    check("Same seed → same # servers", len(obs1.servers) == len(obs2.servers))
    check("Same seed → same # workloads", len(obs1.workloads) == len(obs2.workloads))
    check("Same seed → same budget", obs1.budget_per_hour == obs2.budget_per_hour)

    obs3 = env.reset("random", seed=99)
    different = (len(obs1.servers) != len(obs3.servers) or
                 len(obs1.workloads) != len(obs3.workloads) or
                 obs1.budget_per_hour != obs3.budget_per_hour)
    check("Different seed → different state", different)


def test_spot_eviction():
    print("\n═══ Spot Eviction ═══")
    env = CloudEnvironment()
    env.reset("easy")
    # Provision a spot server and migrate a workload to it
    r = env.step(Action(action_type="provision", instance_type="spot-medium"))
    new_srv = None
    for s in r["observation"].servers:
        if s.is_spot:
            new_srv = s.server_id
    check("Can provision spot instance", new_srv is not None)
    if new_srv:
        check("Spot instance is marked", any(s.is_spot for s in r["observation"].servers if s.server_id == new_srv))


def test_cascading_failures():
    print("\n═══ Cascading Failures ═══")
    env = CloudEnvironment()
    obs = env.reset("hard")
    # wl-api depends on wl-auth. Both start on overloaded srv-001.
    wl_api = next(w for w in obs.workloads if w.workload_id == "wl-api")
    check("wl-api has dependency on wl-auth", "wl-auth" in wl_api.dependencies)

    # Fix wl-auth by migrating it to srv-004
    r = env.step(Action(action_type="migrate", workload_id="wl-auth", target_server_id="srv-004"))
    wl_auth_after = next(w for w in r["observation"].workloads if w.workload_id == "wl-auth")
    check("Migrating wl-auth relieves it", wl_auth_after.current_latency_ms <= wl_auth_after.sla_latency_ms)


def test_traffic_multiplier():
    print("\n═══ Traffic Multiplier ═══")
    m1 = _get_traffic_multiplier(42, 0, "wl-api")
    m2 = _get_traffic_multiplier(42, 1, "wl-api")
    m3 = _get_traffic_multiplier(42, 0, "wl-api")
    check("Traffic multiplier in range", 0.85 <= m1 <= 1.45)
    check("Different steps → different multiplier", m1 != m2)
    check("Same inputs → same multiplier (deterministic)", m1 == m3)
    check("No seed → 1.0", _get_traffic_multiplier(None, 0, "wl-api") == 1.0)


def test_grader_differentiation():
    print("\n═══ Grader Differentiation ═══")
    scores = set()
    env = CloudEnvironment()
    actions_sets = [
        [],  # noop only
        [Action(action_type="resize", server_id="srv-001", instance_type="small")],
        [Action(action_type="resize", server_id="srv-001", instance_type="micro")],
        [Action(action_type="resize", server_id="srv-001", instance_type="medium")],
        [Action(action_type="terminate", server_id="srv-001")],
    ]
    for actions in actions_sets:
        env.reset("easy")
        r = None
        for a in actions:
            r = env.step(a)
        if r is None:
            r = env.step(Action(action_type="noop"))
        # Run to completion
        while not r["done"]:
            r = env.step(Action(action_type="noop"))
        scores.add(r["reward"].score)

    check(f"5 different action sets → {len(scores)} unique scores (need ≥4)",
          len(scores) >= 4, f"Scores: {sorted(scores)}")


if __name__ == "__main__":
    print("=" * 60)
    print("DevOps/FinOps OpenEnv — Test Suite")
    print("=" * 60)

    test_instance_catalog()
    test_cost_efficiency()
    test_performance_score()
    test_penalty()
    test_easy_task()
    test_medium_task()
    test_hard_task()
    test_determinism()
    test_session_management()
    test_procedural_generation()
    test_spot_eviction()
    test_cascading_failures()
    test_traffic_multiplier()
    test_grader_differentiation()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL > 0 else 0)
