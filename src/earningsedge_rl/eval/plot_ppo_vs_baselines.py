import json
import os
import random
import matplotlib.pyplot as plt

RUN_DIR = "runs/sprint3_ppo"
POLICIES = ["ppo", "buy_hold", "avoid_earnings", "flat"]

def load(policy):
    with open(os.path.join(RUN_DIR, f"{policy}_curves.json"), "r") as f:
        return json.load(f)

def main():
    samples = {p: load(p) for p in POLICIES}

    k = min(len(samples["ppo"]), 5)
    idxs = random.sample(range(k), k=min(3, k))

    for idx in idxs:
        plt.figure(figsize=(12, 5))
        for p in POLICIES:
            ep = samples[p][idx]
            plt.plot(ep["equity_curve"], label=f"{p} ({ep['symbol']})")
        plt.title("Equity Curve — PPO vs Baselines (Matched Episodes)")
        plt.xlabel("Step")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
