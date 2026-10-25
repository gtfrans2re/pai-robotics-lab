# RL locomotion — MuJoCo + Stable-Baselines3

Train and watch locomotion policies on MuJoCo control tasks. Two algorithms:
PPO (on-policy, fast per step) and SAC (off-policy, far more sample-efficient on
Humanoid).

## Prerequisites

```bash
./setup/setup_robotics.sh pyenv
source ~/pai-robotics-venv/bin/activate
cd rl
```

## Sanity check first (recommended)

Validate the whole pipeline on an easy env before committing hours to Humanoid.
Edit `ENV_ID = "HalfCheetah-v5"` in `train_ppo.py`, then:

```bash
python train_ppo.py      # climbs ~-330 -> ~+1,080 in ~11 min
python watch_ppo.py
```

## Humanoid

PPO (tuned) — better than default but plateaus:
```bash
python train_ppo.py      # ENV_ID = "Humanoid-v5", ~2h40 for 15M steps
python watch_ppo.py
```

SAC — the stable walker (recommended):
```bash
python train_sac.py      # overnight on CPU (~21h for 3M steps)
python watch_sac.py
python watch_sac.py models/Humanoid-sac_1000000_steps.zip   # peek at a checkpoint mid-run
```

## Results

| Algorithm | Reward | Episode length |
|---|---:|---:|
| PPO (default `[64,64]`) | ~920 | ~180 |
| PPO (tuned `[256,256]`) | ~2,900 | ~310 |
| **SAC** | **~5,500** | **~980** |

Full gait analysis: [../docs/results.md](../docs/results.md).

## Watching / metrics

- **The metric that matters for balance is `ep_len_mean` → 1000**, not just reward
  (Humanoid pays a +5/step "alive" bonus, so early reward looks decent while it
  still falls fast).
- Live curves: `tensorboard --logdir ./tb/` then open the localhost URL.
- Viewer controls: mouse-drag to orbit, Ctrl-drag a body to push it, ESC/Ctrl-C to quit.
- Wayland GL error? Prefix with `MUJOCO_GL=egl` (or `glx`).

## Files

| File | Purpose |
|---|---|
| `train_ppo.py` | Tuned PPO (SB3-zoo Humanoid recipe), `SubprocVecEnv` + `VecNormalize` |
| `watch_ppo.py` | Render a PPO policy (loads the saved `VecNormalize` stats) |
| `train_sac.py` | SAC, single env, net `[256,256]` |
| `watch_sac.py` | Render a SAC policy |

Trained models land in `rl/models/` (gitignored — ship via a GitHub Release or git-lfs).
