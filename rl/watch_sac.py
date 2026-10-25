# Load a trained SAC policy and render it in the MuJoCo viewer.
# Usage: python watch_sac.py [path/to/checkpoint.zip]
import sys
import gymnasium as gym
from stable_baselines3 import SAC

ENV_ID = "Humanoid-v5"
path = sys.argv[1] if len(sys.argv) > 1 else "models/Humanoid-sac_final.zip"

env = gym.make(ENV_ID, render_mode="human")
model = SAC.load(path)
print(f"Loaded {path}  —  ESC or Ctrl-C to quit")
obs, _ = env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
except KeyboardInterrupt:
    print("\nClosed.")
finally:
    env.close()
