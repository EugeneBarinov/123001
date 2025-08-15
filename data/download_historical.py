import argparse
import pandas as pd
from binance.client import Client
import os
from dotenv import load_dotenv
from pathlib import Path

HERE = Path(__file__).resolve().parent


def download_klines(symbol: str, interval: str, start_str: str, limit: int, out_csv: str, api_key: str=None, api_secret: str=None):
    if api_key and api_secret:
        client = Client(api_key, api_secret)
    else:
        client = Client()
    klines = client.get_historical_klines(symbol, interval, start_str, limit=limit)
    cols = ['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base','taker_quote','ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['open_time'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'].astype(int), unit='ms')
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['datetime'] = df['open_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    out_df = df[['datetime','open','high','low','close','volume']]
    out_path = Path(out_csv)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--interval', default='1m')
    parser.add_argument('--start', dest='start_str', default='1 day ago UTC')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--out', dest='out_csv', default='AI/data/historical.csv')
    parser.add_argument('--api_key', default=None)
    parser.add_argument('--api_secret', default=None)
    args = parser.parse_args()
    try:
        load_dotenv(HERE / '.env')
    except Exception:
        pass
    api_key = args.api_key or os.environ.get('BINANCE_API_KEY')
    api_secret = args.api_secret or os.environ.get('BINANCE_API_SECRET')
    df = download_klines(args.symbol, args.interval, args.start_str, args.limit, args.out_csv, api_key, api_secret)
    print('Saved', args.out_csv, 'rows:', len(df))
