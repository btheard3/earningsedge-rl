from __future__ import annotations

import os
import time
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from earningsedge_rl.training.make_env import make_env

RUN_DIR = "runs/sprint3_ppo"

def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    # vectorized env (1 env to start = faster debugging)
    env = DummyVecEnv([lambda: Monitor(make_env(seed=42)())])

    # Conservative, stable PPO defaults
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.0,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=os.path.join(RUN_DIR, "tb"),
    )

    # Keep this small first. If it runs, we scale.
    total_timesteps = 200_000

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, tb_log_name="ppo_run")
    t1 = time.time()

    # Save model
    model_path = os.path.join(RUN_DIR, "ppo_trading_env.zip")
    model.save(model_path)

    # Save run metadata
    meta = {
        "total_timesteps": total_timesteps,
        "elapsed_sec": round(t1 - t0, 2),
        "model_path": model_path,
    }
    with open(os.path.join(RUN_DIR, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", model_path)
    print("Meta:", meta)

if __name__ == "__main__":
    main()
