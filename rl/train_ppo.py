# PPO locomotion training — tuned SB3-zoo Humanoid recipe.
# Swap ENV_ID to "HalfCheetah-v5" for a fast (~11 min) pipeline sanity check.
import os
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

ENV_ID      = "Humanoid-v5"
N_ENVS      = 8                 # parallel sims across CPU cores
TOTAL_STEPS = 15_000_000        # checkpoints let you stop early


def main():
    os.makedirs("models", exist_ok=True)
    env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log="./tb/",
        n_steps=512, batch_size=256, n_epochs=5,
        gamma=0.95, gae_lambda=0.9, clip_range=0.3,
        ent_coef=0.00238306, vf_coef=0.431892, max_grad_norm=2.0,
        learning_rate=3.56987e-5,
        policy_kwargs=dict(
            log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )
    ckpt = CheckpointCallback(save_freq=max(200_000 // N_ENVS, 1),
                              save_path="./models/", name_prefix=ENV_ID)
    model.learn(total_timesteps=TOTAL_STEPS, callback=ckpt, progress_bar=True)
    model.save(f"models/{ENV_ID}_final")
    env.save(f"models/{ENV_ID}_vecnormalize.pkl")
    print("Done. Watch with:  python watch_ppo.py")


if __name__ == "__main__":
    main()
