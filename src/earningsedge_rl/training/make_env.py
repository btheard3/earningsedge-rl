from __future__ import annotations

from typing import Callable
from earningsedge_rl.env.trading_env import TradingEnv

def make_env(seed: int = 0, symbols: str = "random") -> Callable[[], TradingEnv]:
    """
    symbols:
      - "random" -> env samples from its universe
      - "AAPL,MSFT,..." -> restrict training/eval to a fixed list
    """
    sym_list = None
    if symbols and symbols != "random":
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]

    def _init():
        env = TradingEnv(seed=seed, symbols=sym_list)
        return env

    return _init
