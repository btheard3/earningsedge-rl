from __future__ import annotations

import os
import json
import argparse
import numpy as np

from stable_baselines3 import PPO
from earningsedge_rl.env.trading_env import TradingEnv


def run_episode(env: TradingEnv, policy: str, model: PPO | None = None):
    obs, info = env.reset()
    terminated = False
    truncated = False

    equity_curve = [1.0]
    drawdown_curve = [0.0]

    while not (terminated or truncated):
        if policy == "ppo":
            if model is None:
                raise ValueError("PPO policy requires a loaded model.")
            action, _ = model.predict(obs, deterministic=True)

        elif policy == "buy_hold":
            action = 3  # 100%

        elif policy == "flat":
            action = 0  # 0%

        elif policy == "avoid_earnings":
            # eflag is obs[-2] in your env: [..., eflag, exposure]
            eflag = float(obs[-2])
            action = 0 if eflag >= 0.5 else 3

        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, r, terminated, truncated, step_info = env.step(int(action))
        equity_curve.append(step_info["equity"])
        drawdown_curve.append(step_info["drawdown"])

    return {
        "symbol": info.get("symbol"),
        "final_equity": float(equity_curve[-1]),
        "max_drawdown": float(np.max(drawdown_curve)),
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
    }


def summarize(episodes):
    finals = np.array([e["final_equity"] for e in episodes], dtype=float)
    mdds = np.array([e["max_drawdown"] for e in episodes], dtype=float)
    return {
        "mean_final_equity": float(finals.mean()),
        "median_final_equity": float(np.median(finals)),
        "mean_max_drawdown": float(mdds.mean()),
    }


def load_split_symbols(run_dir: str, which: str) -> list[str]:
    """
    Expects runs/<...>/universe_split.json with keys: train, test
    """
    split_path = os.path.join(run_dir, "universe_split.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Missing split file: {split_path}. Run the universe split step first."
        )
    with open(split_path, "r") as f:
        split = json.load(f)

    if which not in split:
        raise KeyError(f"{split_path} missing key '{which}'. Found keys: {list(split.keys())}")

    # normalize to list[str]
    return [str(s).strip() for s in split[which] if str(s).strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True, help="Run directory containing PPO zip and outputs.")
    p.add_argument("--n_episodes", type=int, default=10, help="Episodes per policy.")
    p.add_argument("--device", type=str, default="cpu", help="PPO load device (cpu recommended for eval).")
    p.add_argument("--use_test_symbols", action="store_true", help="Use test symbol pool from universe_split.json.")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "ppo_trading_env.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at: {model_path}")

    model = PPO.load(model_path, device=args.device)

    # If Sprint 4, we evaluate on test symbols only
    symbols = None
    if args.use_test_symbols:
        symbols = load_split_symbols(run_dir, "test")
        print(f"Evaluating on TEST symbols only: {len(symbols)} symbols")

    policies = ["ppo", "buy_hold", "flat", "avoid_earnings"]
    results = {}

    for p in policies:
        episodes = []
        for i in range(args.n_episodes):
            env = TradingEnv(symbols=symbols) if symbols is not None else TradingEnv()
            episodes.append(run_episode(env, p, model=model))
            if (i + 1) % 2 == 0:
                print(f"{p}: completed {i+1}/{args.n_episodes} episodes")

        results[p] = summarize(episodes)

        out_curves = os.path.join(run_dir, f"{p}_curves.json")
        with open(out_curves, "w") as f:
            json.dump(episodes, f)

    out_metrics = os.path.join(run_dir, "metrics.json")
    with open(out_metrics, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()
