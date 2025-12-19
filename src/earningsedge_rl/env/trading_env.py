from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

_PANEL_CACHE: dict[str, pd.DataFrame] = {}

ACTION_LEVELS = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)


class TradingEnv(gym.Env):
    """
    EarningsEdge RL Environment
    - Each episode samples a random ticker 
    - Single-env 
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
        symbols: list[str] | None = None,    # optional train/test ticker pool
        symbol: str | None = None,           # optional force a single ticker
    ):
        super().__init__()

        self.panel_path = panel_path
        self.universe_path = universe_path

        # Store overrides
        self.symbols_override = symbols[:] if symbols is not None else None
        self.symbol_override = symbol

        # Load panel with caching
        if panel_path in _PANEL_CACHE:
            panel = _PANEL_CACHE[panel_path]
        else:
            panel = pd.read_parquet(panel_path)
            _PANEL_CACHE[panel_path] = panel

        self.panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Canonical price column
        if "adj_close" not in self.panel.columns and "close_adjusted" in self.panel.columns:
            self.panel = self.panel.rename(columns={"close_adjusted": "adj_close"})

        # Universe list
        uni = pd.read_csv(universe_path)
        universe_symbols = uni["symbol"].astype(str).tolist()

        # Build by_symbol only for symbols actually present in panel
        panel_symbols = set(self.panel["symbol"].astype(str).unique())

        self.by_symbol: dict[str, pd.DataFrame] = {
            s: self.panel[self.panel["symbol"] == s].sort_values("date").reset_index(drop=True)
            for s in universe_symbols
            if s in panel_symbols
        }

        self.symbols = list(self.by_symbol.keys())

        # Filter overrides to available symbols (prevents KeyErrors later)
        if self.symbols_override is not None:
            self.symbols_override = [s for s in self.symbols_override if s in self.by_symbol]
            if len(self.symbols_override) == 0:
                self.symbols_override = None  # fallback to default

        if self.symbol_override is not None and self.symbol_override not in self.by_symbol:
            self.symbol_override = None  # fallback to default sampling

        self.episode_len = int(episode_len)
        self.warmup = int(warmup)
        self.tc_bps = float(transaction_cost_bps)
        self.dd_penalty = float(dd_penalty)

        self.rng = np.random.default_rng(seed)

        # Observation: [ret_1, ret_5, ret_20, vol_10, vol_20, vol_z, dte, dse, earnings_flag, exposure]
        self.obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Action: discrete exposure levels
        self.action_space = spaces.Discrete(len(ACTION_LEVELS))

        # Episode state
        self.sym: str | None = None
        self.df: pd.DataFrame | None = None
        self.t: int | None = None
        self.start: int | None = None
        self.end: int | None = None

        self.exposure = 0.0
        self.equity = 1.0
        self.peak = 1.0

    def _features_at(self, idx: int) -> np.ndarray:
        assert self.df is not None

        d = self.df
        px = d["adj_close"].to_numpy(dtype=float)

        # returns (assumes idx >= 20; enforced by reset min_start)
        r1 = (px[idx] / px[idx - 1]) - 1.0
        r5 = (px[idx] / px[idx - 5]) - 1.0
        r20 = (px[idx] / px[idx - 20]) - 1.0

        # rolling vol (std of daily returns)
        lo = max(0, idx - 60)
        window_px = px[lo : idx + 1]
        # daily returns in that window
        rets = np.diff(window_px) / (window_px[:-1] + 1e-12)
        if rets.size < 25:
            vol10 = 0.0
            vol20 = 0.0
        else:
            vol10 = float(np.std(rets[-10:]))
            vol20 = float(np.std(rets[-20:]))

        # volume z-score (rolling)
        vol = d["volume"].to_numpy(dtype=float)
        w = vol[lo : idx + 1]
        vol_z = 0.0 if w.size < 10 else float((vol[idx] - w.mean()) / (w.std() + 1e-9))

        dte = float(d["days_to_earnings"].iloc[idx]) if "days_to_earnings" in d.columns else 99999.0
        dse = float(d["days_since_earnings"].iloc[idx]) if "days_since_earnings" in d.columns else 99999.0
        eflag = float(d["is_earnings_window"].iloc[idx]) if "is_earnings_window" in d.columns else 0.0

        obs = np.array(
            [r1, r5, r20, vol10, vol20, vol_z, dte, dse, eflag, float(self.exposure)],
            dtype=np.float32,
        )
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Choose symbol
        if self.symbol_override is not None:
            self.sym = self.symbol_override
        elif self.symbols_override is not None:
            self.sym = str(self.rng.choice(self.symbols_override))
        else:
            self.sym = str(self.rng.choice(self.symbols))

        # Extra safety fallback
        if self.sym not in self.by_symbol:
            self.sym = str(self.rng.choice(self.symbols))

        self.df = self.by_symbol[self.sym]

        # Pick a start with enough history for lookbacks
        n = len(self.df)
        min_start = self.warmup + 20  # need lookbacks up to 20
        max_start = n - self.episode_len - 1
        if max_start <= min_start:
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
        assert self.df is not None
        assert self.t is not None
        assert self.end is not None

        prev_exposure = float(self.exposure)
        self.exposure = float(ACTION_LEVELS[int(action)])

        # transaction cost on exposure change
        turnover = abs(self.exposure - prev_exposure)
        tc = (self.tc_bps / 10000.0) * turnover

        # next day return
        px = self.df["adj_close"].to_numpy(dtype=float)
        r = (px[self.t + 1] / (px[self.t] + 1e-12)) - 1.0

        # portfolio return
        port_r = self.exposure * r - tc

        # update equity + drawdown
        self.equity *= (1.0 + port_r)
        self.peak = max(self.peak, self.equity)
        dd = (self.peak - self.equity) / (self.peak + 1e-12)

        # earnings-aware drawdown penalty
        eflag = 0.0
        if "is_earnings_window" in self.df.columns:
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
            "equity": float(self.equity),
            "exposure": float(self.exposure),
            "turnover": float(turnover),
            "port_r": float(port_r),
            "drawdown": float(dd),
            "t": int(self.t),
        }
        return obs, reward, terminated, truncated, info
