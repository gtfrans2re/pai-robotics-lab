# Load a trained PPO policy and render it in the MuJoCo viewer.
# Usage: python watch_ppo.py [path/to/checkpoint.zip]
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "Humanoid-v5"   # must match train_ppo.py
model_path = sys.argv[1] if len(sys.argv) > 1 else f"models/{ENV_ID}_final.zip"

env = DummyVecEnv([lambda: gym.make(ENV_ID, render_mode="human")])
try:
    env = VecNormalize.load(f"models/{ENV_ID}_vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False
except FileNotFoundError:
    pass

model = PPO.load(model_path)
print(f"Loaded {model_path}  —  ESC or Ctrl-C to quit")
obs = env.reset()
try:
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
except KeyboardInterrupt:
    print("\nClosed.")
finally:
    env.close()
