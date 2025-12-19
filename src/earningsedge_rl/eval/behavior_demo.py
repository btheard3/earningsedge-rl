import matplotlib.pyplot as plt
import numpy as np
from earningsedge_rl.env.trading_env import TradingEnv

def policy_avoid_earnings(obs):
    # obs[-2] is earnings flag in our env
    return 0 if obs[-2] >= 0.5 else 3  # 0% during earnings window else 100%

def main():
    env = TradingEnv()
    obs, info = env.reset()

    eq = [1.0]
    exposure = [0.0]
    eflag = [float(obs[-2])]

    done = False
    while not done:
        a = policy_avoid_earnings(obs)
        obs, r, done, trunc, step_info = env.step(a)
        eq.append(step_info["equity"])
        exposure.append(step_info["exposure"])
        eflag.append(float(obs[-2]) if not done else 0.0)

    # Plot equity + exposure + earnings flag
    x = np.arange(len(eq))

    plt.figure(figsize=(12, 5))
    plt.plot(x, eq)
    plt.title(f"Behavior demo (avoid earnings) — {info['symbol']}")
    plt.xlabel("Step")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.step(x, exposure, where="post")
    plt.title("Exposure over time")
    plt.xlabel("Step")
    plt.ylabel("Exposure")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 2.5))
    plt.step(x, eflag, where="post")
    plt.title("Earnings window flag")
    plt.xlabel("Step")
    plt.ylabel("Flag")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
