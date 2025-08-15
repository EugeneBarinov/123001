import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path


def generate_ohlcv(n: int = 2000, start_price: float = 30000.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    # Геометрическое броуновское движение
    dt = 1.0
    mu = 0.0
    sigma = 0.002
    returns = rng.normal(mu * dt, sigma * np.sqrt(dt), size=n)
    close = start_price * np.exp(np.cumsum(returns))
    # Open как предыдущий close
    open_ = np.r_[close[0], close[:-1]]
    # High/Low вокруг max/min(open, close) с небольшим спредом
    spread = np.abs(rng.normal(0, 0.001, size=n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(50, 500, size=n).astype(float)

    t0 = datetime.utcnow() - timedelta(minutes=n)
    times = [t0 + timedelta(minutes=i) for i in range(n)]
    df = pd.DataFrame({
        'datetime': [t.strftime('%Y-%m-%d %H:%M:%S') for t in times],
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    })
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='AI/data/historical.csv')
    parser.add_argument('--rows', type=int, default=2000)
    args = parser.parse_args()
    df = generate_ohlcv(n=args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('Saved synthetic CSV:', out, 'rows:', len(df)) 