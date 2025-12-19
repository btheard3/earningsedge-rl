from __future__ import annotations

import os
import time
import json
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from earningsedge_rl.training.make_env import make_env


def save_meta(out_dir: str, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(meta)
    meta["timestamp"] = datetime.utcnow().isoformat()
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="random")  # "random" or "AAPL,MSFT,..."
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/sprint4_stretchA/seed_0")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Build env factory. make_env(seed=...) should return a callable that returns an Env.
    env_fn = make_env(seed=args.seed, symbols=args.symbols)
    env = DummyVecEnv([lambda: Monitor(env_fn())])

    # Seed the vectorized env too (helps with reproducibility)
    env.reset()

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
        seed=args.seed,
        tensorboard_log=os.path.join(out_dir, "tb"),
    )

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, tb_log_name="ppo_run", progress_bar=True)
    t1 = time.time()

    model_path = os.path.join(out_dir, "ppo_trading_env.zip")
    model.save(model_path)

    meta = {
        "seed": args.seed,
        "symbols": args.symbols,
        "total_timesteps": args.total_timesteps,
        "elapsed_sec": round(t1 - t0, 2),
        "model_path": model_path,
    }
    save_meta(out_dir, meta)

    print("Saved:", model_path)
    print("Meta:", meta)


if __name__ == "__main__":
    main()
