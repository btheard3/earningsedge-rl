# EarningsEdge RL (v1)

Reinforcement learning project that trains an agent to dynamically control equity exposure around earnings-event risk using daily OHLCV + earnings dates.

## Status
- ✅ Sprint 1 complete: processed equity panel + earnings-distance features + sanity plot
- ⏭ Sprint 2 next: Gym environment (random ticker per episode), baselines, PPO training

## Dataset
Kaggle: US Historical Stock Prices with Earnings Data (daily OHLCV + earnings dates)

## Repo Structure
- `data/processed/` (generated parquet + small artifacts)
- `src/earningsedge_rl/` (pipeline, features, env, training, eval)
- `runs/` (saved metrics + curves for React demo)
- `ui/` (React portfolio page reads saved artifacts)

## Quickstart (dev)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # coming soon
PYTHONPATH=src python -m earningsedge_rl.data.build_processed
PYTHONPATH=src python -m earningsedge_rl.eval.plot_sanity
```

