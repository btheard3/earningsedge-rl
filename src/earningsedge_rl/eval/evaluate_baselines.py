import os
import json
import numpy as np
import pandas as pd

from earningsedge_rl.env.trading_env import TradingEnv
from earningsedge_rl.training.baselines import policy_buy_hold, policy_flat, policy_avoid_earnings

def run_policy(env, policy_fn, episodes=20):
    results = []
    curves = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        eq_curve = [1.0]
        dd_curve = [0.0]

        while not done:
            a = policy_fn(obs)
            obs, r, done, trunc, step_info = env.step(a)
            eq_curve.append(step_info["equity"])
            dd_curve.append(step_info["drawdown"])

        results.append({
            "symbol": info["symbol"],
            "final_equity": eq_curve[-1],
            "max_drawdown": float(np.max(dd_curve)),
        })
        curves.append({"symbol": info["symbol"], "equity_curve": eq_curve, "drawdown_curve": dd_curve})

    return results, curves

def main():
    env = TradingEnv()

    policies = {
        "buy_hold": policy_buy_hold,
        "flat": policy_flat,
        "avoid_earnings": policy_avoid_earnings,
    }

    os.makedirs("runs/sprint2_baselines", exist_ok=True)

    summary = {}
    for name, fn in policies.items():
        res, curves = run_policy(env, fn, episodes=25)
        df = pd.DataFrame(res)
        summary[name] = {
            "mean_final_equity": float(df["final_equity"].mean()),
            "median_final_equity": float(df["final_equity"].median()),
            "mean_max_drawdown": float(df["max_drawdown"].mean()),
        }

        df.to_csv(f"runs/sprint2_baselines/{name}_summary.csv", index=False)
        with open(f"runs/sprint2_baselines/{name}_curves.json", "w") as f:
            json.dump(curves, f)

    with open("runs/sprint2_baselines/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved runs/sprint2_baselines/*")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
