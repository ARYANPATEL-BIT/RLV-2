"""Reward function audit — tests all scenarios the user asked about."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from env import CloudEnvironment, Action

def run_scenario(name, task_id, actions_fn, seed=None):
    env = CloudEnvironment()
    obs = env.reset(task_id, seed=seed)
    rewards = []
    for step in range(obs.max_steps):
        if env.done:
            break
        action = actions_fn(step, obs, env)
        result = env.step(action)
        obs = result["observation"]
        r = result["reward"]
        rewards.append((step, r.score, r.cost_efficiency, r.performance_score, r.penalty))
    final = rewards[-1] if rewards else (0, 0, 0, 0, 0)
    print(f"\n{'='*70}")
    print(f"SCENARIO: {name} (task={task_id})")
    print(f"{'='*70}")
    print(f"{'Step':>4} {'Score':>8} {'CostEff':>8} {'Perf':>8} {'Penalty':>8}")
    print(f"{'-'*4:>4} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
    for step, score, ce, ps, pen in rewards:
        print(f"{step:>4} {score:>8.4f} {ce:>8.4f} {ps:>8.4f} {pen:>8.4f}")
    print(f"\nFINAL SCORE: {final[1]:.4f}")
    return [r[1] for r in rewards]  # all step scores


# ════════════════════════════════════════════════════════════════
# QUESTION 1: Five scenarios on EASY task (cleaner to demonstrate)
# ════════════════════════════════════════════════════════════════
# Easy task:
#   srv-001 xlarge ($1.60): wl-web(0.5c,1r) + wl-api(1c,1.5r) = 1.5cpu,2.5ram
#   Budget: $0.25/hr
#   Goal: resize to smallest type that fits both workloads

print("\n" + "█"*70)
print("QUESTION 1: Five Scenarios on EASY task")
print("█"*70)

# Scenario A: Agent does NOTHING (all noops)
scores_noop = run_scenario(
    "A. All noops (does nothing)", "easy",
    lambda step, obs, env: Action(action_type="noop")
)

# Scenario B: Random actions
import random
random.seed(42)
def random_action(step, obs, env):
    choices = [
        Action(action_type="noop"),
        Action(action_type="provision", instance_type=random.choice(["nano","micro","small","medium"])),
        Action(action_type="resize", server_id="srv-001", instance_type=random.choice(["nano","micro","small","medium","large"])),
    ]
    return random.choice(choices)

scores_random = run_scenario("B. Random actions", "easy", random_action)

# Scenario C: Good first half, bad second half
def good_then_bad(step, obs, env):
    if step == 0: return Action(action_type="resize", server_id="srv-001", instance_type="small")  # good!
    if step == 1: return Action(action_type="resize", server_id="srv-001", instance_type="xlarge")  # undo it!
    return Action(action_type="noop")

scores_good_bad = run_scenario("C. Good first half, bad second half", "easy", good_then_bad)

# Scenario D: Perfect decisions
def optimal_actions(step, obs, env):
    if step == 0: return Action(action_type="resize", server_id="srv-001", instance_type="small")  # 2cpu/4ram fits 1.5cpu/2.5ram
    return Action(action_type="noop")

scores_optimal = run_scenario("D. Perfect decision (resize to small)", "easy", optimal_actions)

# Scenario E: Repeated same action in a loop
scores_loop = run_scenario(
    "E. Same action repeated (loop)", "easy",
    lambda step, obs, env: Action(action_type="resize", server_id="srv-001", instance_type="small")
)


# ════════════════════════════════════════════════════════════════
# Also show HARD task scenarios for completeness
# ════════════════════════════════════════════════════════════════

print("\n" + "█"*70)
print("ADDITIONAL: Key Scenarios on HARD task")
print("█"*70)

# Hard noop
scores_hard_noop = run_scenario(
    "HARD: All noops", "hard",
    lambda step, obs, env: Action(action_type="noop")
)

# Hard optimal
# Hard task initial state:
#   srv-001 medium (4cpu,8ram,$0.40): wl-auth(2c,3r) + wl-api(2c,2r) ← OVERLOADED 100% CPU, both breach SLA
#   srv-002 large (8cpu,16ram,$0.80): wl-db(2c,4r) ← very underloaded
#   srv-003 nano (1cpu,1ram,$0.05): wl-search(1c,2r) ← MASSIVE overload, breaching
#   srv-004 medium (4cpu,8ram,$0.40): empty ← idle waste
#   srv-005 nano (1cpu,1ram,$0.05): wl-metrics(0.5c,0.5r) ← OK
#   wl-ml-jobs (3c,4r): ORPHANED
#   Budget: $1.20/hr; Current: $1.70/hr
# Optimal strategy:
#   1. Move wl-api off overloaded srv-001 → idle srv-004 (medium fits 2c,2r easily)
#   2. Move wl-search off undersized srv-003 → srv-004 (3c,4r total on 4c,8r medium)
#   3. Assign orphaned wl-ml-jobs to srv-002 (5c,8r total on 8c,16r large)
#   4. Terminate empty srv-003 (save $0.05)
#   5. Resize srv-001 to small (wl-auth 2c,3r fits on 2c,4r small, $0.20)
#   6. Move wl-metrics to srv-004 (3.5c,4.5r on 4c,8r medium — still fits)
#   7. Terminate empty srv-005 (save $0.05)
# Result: srv-001 small($0.20) + srv-002 large($0.80) + srv-004 medium($0.40) = $1.40
#   Still over budget... need to also resize srv-002
#   Actually: wl-db(2c,4r) + wl-ml-jobs(3c,4r) = 5c,8r → medium(4c,8r) won't fit. Need large.
#   So we can't really save on srv-002 without compute-opt.
# Better: just keep srv-002 large, save elsewhere.
# Even better: check if wl-api+wl-search on srv-004 medium is not overloaded
#   wl-api(2c,2r,800disk,0.3net) + wl-search(1c,2r,1000disk,0.2net) = 3c,4r,1800disk,0.5net on medium(4c,8r,5000disk,2.0net) → fine
# So: srv-001 small($0.20) + srv-002 large($0.80) + srv-004 medium($0.40) = $1.40, budget $1.20. Still over.
# Need to resize srv-002: wl-db(2c,4r) + wl-ml-jobs(3c,4r) = 5c,8r, disk: 5000 → need compute-opt(8c,8r,5000disk,$0.60)
# Total: $0.20 + $0.60 + $0.40 = $1.20 exactly! But disk might be tight on compute-opt...
# Let's try it.

def hard_optimal(step, obs, env):
    if step == 0: return Action(action_type="migrate", workload_id="wl-api", target_server_id="srv-004")
    if step == 1: return Action(action_type="migrate", workload_id="wl-search", target_server_id="srv-004")
    if step == 2: return Action(action_type="migrate", workload_id="wl-ml-jobs", target_server_id="srv-002")
    if step == 3: return Action(action_type="terminate", server_id="srv-003")
    if step == 4: return Action(action_type="resize", server_id="srv-001", instance_type="small")
    if step == 5: return Action(action_type="migrate", workload_id="wl-metrics", target_server_id="srv-004")
    if step == 6: return Action(action_type="terminate", server_id="srv-005")
    if step == 7: return Action(action_type="resize", server_id="srv-002", instance_type="compute-opt")
    return Action(action_type="noop")

scores_hard_opt = run_scenario("HARD: Near-optimal", "hard", hard_optimal)


# ════════════════════════════════════════════════════════════════
# QUESTION 2: Dense vs Sparse?
# ════════════════════════════════════════════════════════════════

print("\n" + "█"*70)
print("QUESTION 2: Is the reward DENSE or SPARSE?")
print("█"*70)

print(f"\nOptimal EASY scenario — reward at EVERY step:")
for i, s in enumerate(scores_optimal):
    changed = "← CHANGED" if i > 0 and s != scores_optimal[i-1] else ""
    print(f"  Step {i}: {s:.4f} {changed}")

print(f"\nHard optimal scenario — reward at EVERY step:")
for i, s in enumerate(scores_hard_opt):
    changed = "← CHANGED" if i > 0 and s != scores_hard_opt[i-1] else ""
    print(f"  Step {i}: {s:.4f} {changed}")

all_same_easy = len(set(scores_optimal)) == 1
all_same_hard = len(set(scores_hard_opt)) == 1
print(f"\nEasy: Unique values across steps: {len(set(scores_optimal))}")
print(f"Hard: Unique values across steps: {len(set(scores_hard_opt))}")

if not all_same_hard:
    print("VERDICT: ✅ DENSE — agent gets changing signal at multiple steps (hard task: " + str(len(set(scores_hard_opt))) + " unique values)")
else:
    print("VERDICT: ⚠️ NEEDS REVIEW")


# ════════════════════════════════════════════════════════════════
# QUESTION 3: Partial credit — 5 quality levels  
# Using EASY task for clean demonstration
# ════════════════════════════════════════════════════════════════

print("\n" + "█"*70)
print("QUESTION 3: Partial Credit — 5 quality levels (easy task)")
print("█"*70)

def get_final_score(task_id, actions_fn):
    """Run a full episode and return the final reward object."""
    env = CloudEnvironment()
    obs = env.reset(task_id)
    r = None
    for step in range(obs.max_steps):
        if env.done:
            break
        action = actions_fn(step, obs, env)
        result = env.step(action)
        r = result
    return r["reward"] if r else None

# Level 1: Terrible — terminate server with workloads (destructive)
r1 = get_final_score("easy", lambda step, obs, env:
    Action(action_type="terminate", server_id="srv-001") if step == 0 else
    Action(action_type="noop"))

# Level 2: Bad — do absolutely nothing (xlarge stays at $1.60/hr vs $0.25 budget)
r2 = get_final_score("easy", lambda step, obs, env: Action(action_type="noop"))

# Level 3: Mediocre — resize to large ($0.80/hr, still 3.2x over budget)
def mediocre(step, obs, env):
    if step == 0: return Action(action_type="resize", server_id="srv-001", instance_type="large")
    return Action(action_type="noop")
r3 = get_final_score("easy", mediocre)

# Level 4: Good — resize to medium ($0.40/hr, 1.6x over budget)
def good(step, obs, env):
    if step == 0: return Action(action_type="resize", server_id="srv-001", instance_type="medium")
    return Action(action_type="noop")
r4 = get_final_score("easy", good)

# Level 5: Excellent — resize to small ($0.20/hr, UNDER budget!)
def excellent(step, obs, env):
    if step == 0: return Action(action_type="resize", server_id="srv-001", instance_type="small")
    return Action(action_type="noop")
r5 = get_final_score("easy", excellent)

scores_5 = [r1.score, r2.score, r3.score, r4.score, r5.score]
labels = [
    "Terrible (destructive term)",
    "Bad (all noops, $1.60/hr)", 
    "Mediocre (resize→large $0.80)",
    "Good (resize→medium $0.40)",
    "Excellent (resize→small $0.20)",
]

print(f"\n{'Level':<30} {'Score':>8} {'CostEff':>8} {'Perf':>8} {'Penalty':>8}")
print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for label, r in zip(labels, [r1, r2, r3, r4, r5]):
    print(f"{label:<30} {r.score:>8.4f} {r.cost_efficiency:>8.4f} {r.performance_score:>8.4f} {r.penalty:>8.4f}")

unique = len(set(scores_5))
print(f"\nUnique scores: {unique}/5")
monotonic = all(scores_5[i] <= scores_5[i+1] for i in range(4))
strictly_increasing = all(scores_5[i] < scores_5[i+1] for i in range(4))
print(f"Monotonically increasing: {monotonic}")
print(f"Strictly increasing: {strictly_increasing}")
if unique >= 4 and monotonic:
    print("VERDICT: ✅ PASS — distinct scores, correctly ordered")
elif unique == 5:
    print("VERDICT: ⚠️ 5 distinct scores but NOT monotonically increasing — check reward function")
else:
    print(f"VERDICT: ❌ FAIL — only {unique} unique scores out of 5")


# ════════════════════════════════════════════════════════════════
# QUESTION 4: Loop penalty
# ════════════════════════════════════════════════════════════════

print("\n" + "█"*70)
print("QUESTION 4: Loop Penalty")
print("█"*70)

# Agent that loops (same resize over and over)
env2 = CloudEnvironment()
env2.reset("easy")
for _ in range(5):
    if not env2.done:
        r = env2.step(Action(action_type="resize", server_id="srv-001", instance_type="small"))
s_loop = r["reward"].score
pen_loop = r["reward"].penalty

# Agent that explores (resize once, then noops)
env2 = CloudEnvironment()
env2.reset("easy")
env2.step(Action(action_type="resize", server_id="srv-001", instance_type="small"))
for _ in range(4):
    if not env2.done:
        r = env2.step(Action(action_type="noop"))
s_explore = r["reward"].score
pen_explore = r["reward"].penalty

print(f"\n{'Strategy':<30} {'Score':>8} {'Penalty':>8}")
print(f"{'-'*30} {'-'*8} {'-'*8}")
print(f"{'Loop (same resize x5)':<30} {s_loop:>8.4f} {pen_loop:>8.4f}")
print(f"{'Explore (resize + noops)':<30} {s_explore:>8.4f} {pen_explore:>8.4f}")
print(f"\nScore difference: {s_explore - s_loop:+.4f} (explore is better)")
print(f"Penalty difference: {pen_loop - pen_explore:+.4f} (loop is worse)")

if s_explore > s_loop and pen_loop > pen_explore:
    print("VERDICT: ✅ Loop penalty works — exploring beats looping")
else:
    print("VERDICT: ❌ Loop penalty NOT working")
