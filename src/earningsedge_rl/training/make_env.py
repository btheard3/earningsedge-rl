from __future__ import annotations

from typing import Callable, Optional, List, Union

from earningsedge_rl.env.trading_env import TradingEnv


def _normalize_symbols(symbols: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    if symbols is None:
        return None
    if isinstance(symbols, list):
        return [str(s).strip() for s in symbols if str(s).strip()]
    if isinstance(symbols, str):
        return [s.strip() for s in symbols.split(",") if s.strip()]
    # last-resort
    return [str(symbols).strip()]


def make_env(
    seed: int = 0,
    symbols: Optional[Union[str, List[str]]] = None,  # train/test pool
    symbol: Optional[str] = None,                     # force single ticker
) -> Callable[[], TradingEnv]:
    sym_list = _normalize_symbols(symbols)

    def _init() -> TradingEnv:
        env = TradingEnv(seed=seed, symbols=sym_list, symbol=symbol)
        return env

    return _init

