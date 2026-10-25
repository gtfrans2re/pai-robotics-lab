# Results & analysis

All runs trained **entirely on the CPU** of an Intel Core Ultra 5 125H (Meteor
Lake, 14C/18T), 32 GB, Ubuntu 24.04 — no GPU acceleration.

## HalfCheetah-v5 (pipeline sanity check)

PPO, 8 parallel envs, 1 M steps, ~11 minutes:

- `ep_rew_mean`: **−331 → +1,080**
- Textbook PPO curve: flat-negative while flailing, sharp climb through zero
  around iteration ~21 (the moment it discovers forward motion), then steady gains.

## Humanoid-v5

| Run | Algorithm | Config | Reward | Ep. length | Wall-clock |
|---|---|---|---:|---:|---:|
| 1 | PPO | default net `[64,64]`, generic HP | ~920 | ~180 | ~11 min / 1 M |
| 2 | PPO | tuned net `[256,256]`, SB3-zoo HP | ~2,900 | ~310 | ~2h40 / 15 M |
| 3 | **SAC** | net `[256,256]`, default HP | **~5,500** | **~980** | ~21 h / 3 M |

Reference: a stable walker sits at ep_len ≈ 1000 and reward ≈ 5,000–6,000.

### Why PPO plateaued (run 2)

The tuned PPO run tripled the reward but stalled around 2,900 with telltale
optimization stress:

- `approx_kl` climbed to **~0.19** (healthy is < 0.03) — the policy was taking
  over-large steps.
- `std` collapsed to **~0.115** — the policy became nearly deterministic and
  stopped exploring.

This is the classic PPO-on-Humanoid wall: more steps don't help once the policy
over-sharpens. SAC (off-policy, with entropy regularization and a replay buffer)
sidesteps it.

## Gait analysis — PPO vs SAC

From the side-by-side recording (`assets/humanoid_ppo_vs_sac.gif`), the two
algorithms converged to **qualitatively different optima**:

**PPO (left) — fast but a controlled fall forward**
- Torso pitched sharply forward, often hunched low; starts almost folded over.
- Essentially a forward-diving lope: converts "lean past the balance point and
  catch yourself" into speed — covers a lot of ground fast.
- Arms aren't used for balance; they hang or flail.
- High forward velocity, poor posture, living on the edge of falling (hence the
  short ~310 episode length).

**SAC (right) — upright, balance-first**
- Tall, upright torso held over the feet throughout.
- Arms held out wide as active counterbalance — a learned "tightrope" strategy
  and a strong sign of a real balance controller.
- Wider, planted stance; recovers small perturbations in place rather than by
  falling forward.
- Travels forward more slowly but stays up for full episodes (~980).

**Takeaway:** not "bad vs good" of one gait — two different optima. PPO maximized
raw forward progress by sacrificing posture; SAC maximized *sustained* return by
prioritizing balance. The 2,900 vs 5,500 reward gap is visible in the posture.

### Known trade-off

SAC's steadiness comes partly at the cost of forward *speed* — stable but
conservative. Getting both (stable **and** fast) is a reward-shaping problem
(raise the forward-velocity weight, or lower the control-cost penalty) plus more
steps — not an algorithm problem.
