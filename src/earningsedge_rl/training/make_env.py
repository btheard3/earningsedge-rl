from __future__ import annotations
from typing import Callable
from earningsedge_rl.env.trading_env import TradingEnv

def make_env(seed: int = 0) -> Callable[[], TradingEnv]:
    def _init():
        env = TradingEnv()
        env.reset(seed=seed)
        return env
    return _init
