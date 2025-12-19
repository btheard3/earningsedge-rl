import os, json
import numpy as np

RUN_DIR = "runs/sprint3_ppo"
POLICIES = ["ppo", "buy_hold", "flat", "avoid_earnings"]

def summarize(episodes):
    finals = np.array([e["final_equity"] for e in episodes], dtype=float)
    mdds = np.array([e["max_drawdown"] for e in episodes], dtype=float)
    return {
        "mean_final_equity": float(finals.mean()),
        "median_final_equity": float(np.median(finals)),
        "mean_max_drawdown": float(mdds.mean()),
    }

def main():
    results = {}
    for p in POLICIES:
        path = os.path.join(RUN_DIR, f"{p}_curves.json")
        if not os.path.exists(path):
            print("Missing:", path)
            continue
        with open(path, "r") as f:
            episodes = json.load(f)
        results[p] = summarize(episodes)

    out = os.path.join(RUN_DIR, "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print("Saved:", out)

if __name__ == "__main__":
    main()
