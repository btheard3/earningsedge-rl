from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
import random
from gymnasium import spaces

_PANEL_CACHE = {}

ACTION_LEVELS = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)

class TradingEnv(gym.Env):
    """
    EarningsEdge RL Environment
    - Each episode samples a random ticker (Option 2)
    - Single-env (Option B)
    - Agent controls exposure level: 0/25/50/100%
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel_path: str = "data/processed/panel.parquet",
        universe_path: str = "data/processed/universe_top200.csv",
        episode_len: int = 252,
        warmup: int = 30,
        transaction_cost_bps: float = 5.0,   # 5 bps per exposure change
        dd_penalty: float = 0.10,            # drawdown penalty weight
        seed: int | None = None,
        symbols: list[str] | None = None,    # <-- NEW (train/test tickers list)
        symbol: str | None = None,           # <-- NEW (force a single ticker)
    ):
        super().__init__()

        self.symbols_override = symbols
        self.symbol_override = symbol

        if panel_path in _PANEL_CACHE:
            self.panel = _PANEL_CACHE[panel_path]
        else:
            self.panel = pd.read_parquet(panel_path)
        _PANEL_CACHE[panel_path] = self.panel

        self.panel = self.panel.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Use adj_close canonical name
        if "adj_close" not in self.panel.columns and "close_adjusted" in self.panel.columns:
            self.panel = self.panel.rename(columns={"close_adjusted": "adj_close"})

        # Universe (default symbol pool)
        uni = pd.read_csv(universe_path)
        self.symbols = uni["symbol"].astype(str).tolist()

        # Pre-split by symbol for fast sampling
        panel_symbols = set(self.panel["symbol"].unique())

        self.by_symbol = {
            s: self.panel[self.panel["symbol"] == s].sort_values("date").reset_index(drop=True)
            for s in self.symbols
            if s in panel_symbols
        }

        self.symbols = list(self.by_symbol.keys())

        self.episode_len = episode_len
        self.warmup = warmup
        self.tc_bps = transaction_cost_bps
        self.dd_penalty = dd_penalty

        self.rng = np.random.default_rng(seed)

        # Observation: [ret_1, ret_5, ret_20, vol_10, vol_20, vol_z, dte, dse, earnings_flag, exposure]
        self.obs_dim = 10
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Action: discrete exposure levels
        self.action_space = spaces.Discrete(len(ACTION_LEVELS))

        # Episode state
        self.sym = None
        self.df = None
        self.t = None
        self.start = None
        self.end = None

        self.exposure = 0.0
        self.equity = 1.0
        self.peak = 1.0

    def _features_at(self, idx: int) -> np.ndarray:
        d = self.df

        # returns
        px = d["adj_close"].to_numpy()
        r1 = (px[idx] / px[idx-1]) - 1.0
        r5 = (px[idx] / px[idx-5]) - 1.0
        r20 = (px[idx] / px[idx-20]) - 1.0

        # rolling vol (std of daily returns)
        rets = np.diff(px[max(0, idx-60):idx+1]) / px[max(0, idx-60):idx]
        # protect length
        if rets.size < 25:
            vol10 = vol20 = 0.0
        else:
            vol10 = float(np.std(rets[-10:]))
            vol20 = float(np.std(rets[-20:]))

        # volume z-score (rolling)
        vol = d["volume"].to_numpy(dtype=float)
        w = vol[max(0, idx-60):idx+1]
        vol_z = 0.0 if w.size < 10 else float((vol[idx] - w.mean()) / (w.std() + 1e-9))

        dte = float(d["days_to_earnings"].iloc[idx]) if "days_to_earnings" in d.columns else 99999.0
        dse = float(d["days_since_earnings"].iloc[idx]) if "days_since_earnings" in d.columns else 99999.0
        eflag = float(d["is_earnings_window"].iloc[idx]) if "is_earnings_window" in d.columns else 0.0

        obs = np.array([r1, r5, r20, vol10, vol20, vol_z, dte, dse, eflag, self.exposure], dtype=np.float32)
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.symbol_override is not None:
            # force a single ticker (useful for demos / fixed evaluation)
            self.sym = self.symbol_override
        elif self.symbols_override is not None:
            # multi-asset training/eval pool (train/test splits)
            self.sym = random.choice(self.symbols_override)
        else:
            # default: sample from universe (your current behavior)
            self.sym = self.rng.choice(self.symbols)

        self.df = self.by_symbol[self.sym]

        # pick a random start with enough history for warmup + episode
        n = len(self.df)
        min_start = self.warmup + 20  # need lookbacks up to 20
        max_start = n - self.episode_len - 1
        if max_start <= min_start:
            # fallback: use earliest possible
            self.start = min_start
        else:
            self.start = int(self.rng.integers(min_start, max_start))

        self.end = self.start + self.episode_len
        self.t = self.start

        self.exposure = 0.0
        self.equity = 1.0
        self.peak = 1.0

        obs = self._features_at(self.t)
        info = {"symbol": self.sym, "start_idx": self.start}
        return obs, info

    def step(self, action: int):
        prev_exposure = self.exposure
        self.exposure = float(ACTION_LEVELS[action])

        # transaction cost on exposure change
        turnover = abs(self.exposure - prev_exposure)
        tc = (self.tc_bps / 10000.0) * turnover

        # next day return
        px = self.df["adj_close"].to_numpy()
        r = (px[self.t + 1] / px[self.t]) - 1.0

        # portfolio return
        port_r = self.exposure * r - tc

        # update equity + drawdown
        self.equity *= (1.0 + port_r)
        self.peak = max(self.peak, self.equity)
        dd = (self.peak - self.equity) / (self.peak + 1e-12)

        # reward = return - drawdown penalty
        # earnings-aware drawdown penalty
        eflag = 0.0
        if self.df is not None and "is_earnings_window" in self.df.columns:
            eflag = float(self.df["is_earnings_window"].iloc[self.t])

        dd_weight = self.dd_penalty * (3.0 if eflag == 1.0 else 1.0)
        reward = float(port_r - dd_weight * dd)

        # advance time
        self.t += 1
        terminated = self.t >= self.end
        truncated = False

        obs = self._features_at(self.t) if not terminated else np.zeros((self.obs_dim,), dtype=np.float32)
        info = {
            "symbol": self.sym,
            "equity": self.equity,
            "exposure": self.exposure,
            "turnover": turnover,
            "port_r": port_r,
            "drawdown": dd,
            "t": self.t,
        }
        return obs, reward, terminated, truncated, info
