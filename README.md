# EarningsEdge RL

**Reinforcement Learning for Earnings-Aware Equity Exposure**

> A reproducible reinforcement learning system that trains an agent to dynamically control equity exposure around earnings events, benchmarked against buy-and-hold, earnings-avoidance, and flat exposure baselines.

## 1. Problem Statement (Why This Exists)

Earnings events introduce **discrete, asymmetric risk** that traditional buy-and-hold strategies ignore.

Most retail strategies either:

- Stay fully invested and absorb volatility, or

- Avoid earnings entirely and sacrifice upside.

Goal:
Train an RL agent to adaptively scale exposure around earnings events using historical context, volatility, and price dynamics — then rigorously evaluate whether it improves outcomes versus simple baselines.

## 2. Approach (What Was Built)

**Core Idea**

Model earnings-aware trading as a sequential decision problem:

- State: price history, volatility, earnings proximity, engineered features

- Action: scale exposure (not direction)

- Reward: equity growth with drawdown sensitivity

Methods

- Environment: custom Gym-style environment with random ticker per episode

- Agent: PPO (Proximal Policy Optimization)

- Baselines:

    - Buy & Hold

    - Avoid Earnings

    - Flat Exposure

## 3. Experimental Design (How It Was Tested)

**Sprints**
Sprint	    Description
Sprint 2	Baselines + environment sanity checks
Sprint 3	PPO training
Sprint 4	Cross-ticker generalization
Sprint 5	Long-horizon training + matched episode evaluation

**Evaluation Guarantees**

- PPO and baselines evaluated on matched episodes

- No look-ahead leakage

- Metrics computed offline, not in the UI

## 4. Results (What Happened)

Results are surfaced through a **React dashboard** built entirely from exported artifacts.

**Key Views**

- **Overview** – experiment scope, metrics, interpretation

- **PPO vs Baselines** – equity curves and drawdown comparison

- **Error Analysis** – where and why the agent failed

- **Sprint 5 Plan** – long-train conclusions and next steps

All metrics are sourced from:
```swift
ui/earningsedge-dashboard/public/artifacts/{sprint4|sprint5}
```

No notebooks required to review results.

## 5. Failure Analysis (Why This Is Honest)

Sprint 5 includes **symbol-level failure attribution**, including:

- Failure rate per ticker

- Peak-to-trough drawdown diagnostics

- Structured failure reasons and flags (e.g. `HARD_DRAWDOWN`, `LATE_CRASH`)

This avoids the common RL pitfall of reporting only averages.

## 6. Reproducibility (Runs on Any Machine)

**Requirements**

- Python 3.10+

- Node 18+

Quickstart (Dashboard Only)
```bash
cd ui/earningsedge-dashboard
npm install
npm run dev:ready
```

This command:

1. Builds evaluation CSVs

2. Publishes UI artifacts

3. Starts the dashboard

No notebooks. No manual steps.

## 7. Repository Structure
```perl
runs/
  sprint*_*/
    *_curves.json
    metrics.json
    summary_table.csv
    symbol_failure_summary.csv

scripts/
  build_eval_csvs.py        # canonical evaluation logic
  export_ui_artifacts.py    # publishes artifacts to UI

ui/earningsedge-dashboard/
  public/artifacts/         # UI reads only from here
  src/                      # React + Vite dashboard
```

## 8. Design Principles

- **Separation of concerns**
Training ≠ Evaluation ≠ Visualization

- **Deterministic UI**
UI never computes metrics

- **Honest reporting**
Failures are first-class outputs

## 9. What This Project Demonstrates

- Reinforcement learning applied to real financial constraints

- Experimental discipline (matched episodes, baselines)

- Production-style evaluation pipelines

- Frontend + backend integration

- Clear communication of uncertainty and failure modes

## 10. Future Extensions (Optional)

- Risk-aware reward shaping

- Transaction cost modeling

- Regime-conditioned policies

- Multi-asset portfolios


## Dataset
Kaggle: US Historical Stock Prices with Earnings Data (daily OHLCV + earnings dates)


