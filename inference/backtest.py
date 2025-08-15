import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().fillna(0.0)


def run_naive_strategy(df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    # signals: BUY -> 1, SELL -> 0 (flat / short disabled)
    pos = signals.map({"BUY": 1, "SELL": 0}).fillna(0).astype(float)
    ret = compute_returns(df)
    strat_ret = pos.shift(1).fillna(0) * ret
    equity = (1 + strat_ret).cumprod()
    return pd.DataFrame({"ret": strat_ret, "equity": equity})


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 365 * 24 * 60) -> float:
    # For 1-min bars by default
    excess = returns - risk_free / periods_per_year
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return (excess.mean() / std) * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())
