import json
import os
import random
import matplotlib.pyplot as plt

RUN_DIR = "runs/sprint2_baselines"
POLICIES = ["buy_hold", "flat", "avoid_earnings"]

def load_curves(policy: str):
    path = os.path.join(RUN_DIR, f"{policy}_curves.json")
    with open(path, "r") as f:
        return json.load(f)

def plot_equity(policy: str, k: int = 3):
    curves = load_curves(policy)
    picks = random.sample(curves, k=min(k, len(curves)))

    plt.figure(figsize=(12, 5))
    for item in picks:
        eq = item["equity_curve"]
        plt.plot(eq, label=item["symbol"])
    plt.title(f"{policy}: equity curves (sample episodes)")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_drawdown(policy: str, k: int = 3):
    curves = load_curves(policy)
    picks = random.sample(curves, k=min(k, len(curves)))

    plt.figure(figsize=(12, 5))
    for item in picks:
        dd = item["drawdown_curve"]
        plt.plot(dd, label=item["symbol"])
    plt.title(f"{policy}: drawdown curves (sample episodes)")
    plt.xlabel("Step")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    for p in POLICIES:
        plot_equity(p, k=3)
        plot_drawdown(p, k=3)

if __name__ == "__main__":
    main()
