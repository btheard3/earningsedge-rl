from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from earningsedge_rl.training.make_env import make_env
from earningsedge_rl.training.universe_split import split_universe, save_split, load_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="runs/sprint4_generalization")
    parser.add_argument("--universe_path", default="data/processed/universe_top200.csv")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--split_path", default=None, help="If provided, load split JSON instead of creating one.")
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    # 1) split
    split_path = args.split_path or os.path.join(args.run_dir, "universe_split.json")
    if args.split_path and os.path.exists(args.split_path):
        split = load_split(args.split_path)
    elif os.path.exists(split_path):
        split = load_split(split_path)
    else:
        split = split_universe(
            universe_path=args.universe_path,
            test_frac=args.test_frac,
            seed=args.seed,
            max_train=args.max_train,
            max_test=args.max_test,
        )
        save_split(split, split_path)

    print(f"Train symbols: {len(split.train)} | Test symbols: {len(split.test)}")
    print(f"Saved split: {split_path}")

    # 2) env: TRAIN ONLY
    env = DummyVecEnv([lambda: Monitor(make_env(seed=args.seed, symbols=split.train)())])

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
        tensorboard_log=os.path.join(args.run_dir, "tb"),
        device="auto",
    )

    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps, tb_log_name="ppo_train")
    t1 = time.time()

    model_path = os.path.join(args.run_dir, "ppo_trading_env.zip")
    model.save(model_path)

    meta = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_timesteps": args.total_timesteps,
        "elapsed_sec": round(t1 - t0, 2),
        "seed": args.seed,
        "model_path": model_path,
        "split_path": split_path,
        "train_symbols_count": len(split.train),
        "test_symbols_count": len(split.test),
    }
    with open(os.path.join(args.run_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", model_path)
    print("Meta:", meta)


if __name__ == "__main__":
    main()
