from __future__ import annotations

import os
import json
import numpy as np

from stable_baselines3 import PPO
from earningsedge_rl.env.trading_env import TradingEnv

RUN_DIR = "runs/sprint3_ppo"

def run_episode(env: TradingEnv, policy: str, model: PPO | None = None):
    obs, info = env.reset()
    done = False

    equity_curve = [1.0]
    drawdown_curve = [0.0]

    while not done:
        if policy == "ppo":
            action, _ = model.predict(obs, deterministic=True)
        elif policy == "buy_hold":
            action = 3  # 100%
        elif policy == "flat":
            action = 0  # 0%
        elif policy == "avoid_earnings":
            # earnings flag is last/near-last in obs; adjust if your env differs
            eflag = float(obs[-2])
            action = 0 if eflag >= 0.5 else 3
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, r, done, trunc, step_info = env.step(int(action))
        equity_curve.append(step_info["equity"])
        drawdown_curve.append(step_info["drawdown"])

    return {
        "symbol": info["symbol"],
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

def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    model_path = os.path.join(RUN_DIR, "ppo_trading_env.zip")
    model = PPO.load(model_path, device="cpu")

    policies = ["ppo", "buy_hold", "flat", "avoid_earnings"]
    results = {}

    # run a small evaluation first (fast). scale later.
    n_episodes = 10 

    for p in policies:
        env = TradingEnv()  # new env per policy batch
        episodes = []
        for i in range(n_episodes):
            env = TradingEnv()
            episodes.append(run_episode(env, p, model=model))
            if (i + 1) % 2 == 0:
                print(f"{p}: completed {i+1}/{n_episodes} episodes")
        results[p] = summarize(episodes)

        with open(os.path.join(RUN_DIR, f"{p}_curves.json"), "w") as f:
            json.dump(episodes, f)

    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved {p}_curves.json", flush=True)
    print("Saved to:", RUN_DIR)

if __name__ == "__main__":
    main()
