from __future__ import annotations
import os, json
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
            action = 3
        elif policy == "flat":
            action = 0
        elif policy == "avoid_earnings":
            eflag = float(obs[-2])  # adjust if needed
            action = 0 if eflag >= 0.5 else 3
        else:
            raise ValueError(policy)

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

    # Only load PPO if needed (not needed for avoid_earnings)
    model_path = os.path.join(RUN_DIR, "ppo_trading_env.zip")
    model = PPO.load(model_path, device="cpu")

    policy = "avoid_earnings"
    n_episodes = 10

    episodes = []
    for i in range(n_episodes):
        env = TradingEnv()
        episodes.append(run_episode(env, policy, model=model))
        print(f"{policy}: {i+1}/{n_episodes}", flush=True)

    out_path = os.path.join(RUN_DIR, f"{policy}_curves.json")
    with open(out_path, "w") as f:
        json.dump(episodes, f)

    print("Saved:", out_path)
    print("Summary:", summarize(episodes))

if __name__ == "__main__":
    main()
