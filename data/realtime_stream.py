import websocket, json, time
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAVE_FILE = HERE / 'realtime.csv'

def on_message(ws, message):
    msg = json.loads(message)
    k = msg.get('k', {})
    if not k:
        return
    if not k.get('x', False):
        return
    row = {
        'datetime': pd.to_datetime(k['t'], unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
        'open': float(k['o']), 'high': float(k['h']),
        'low': float(k['l']), 'close': float(k['c']),
        'volume': float(k['v'])
    }
    df = pd.DataFrame([row])
    header = not SAVE_FILE.exists()
    df.to_csv(SAVE_FILE, mode='a', header=header, index=False)

def start_stream(symbol='btcusdt', interval='1m'):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    while True:
        try:
            ws.run_forever()
        except Exception as e:
            print('Websocket error:', e)
            time.sleep(5)

if __name__ == '__main__':
    start_stream()
