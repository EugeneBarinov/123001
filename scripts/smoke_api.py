import requests
import json

BASE = "http://127.0.0.1:8000"
CSV = "AI/data/historical.csv"

def main():
    r = requests.get(f"{BASE}/health", timeout=10)
    print("/health:", r.status_code, r.text)

    r = requests.post(f"{BASE}/predict", json={"csv_path": CSV}, timeout=60)
    print("/predict:", r.status_code, r.text)

    r = requests.post(f"{BASE}/backtest", json={"csv_path": CSV}, timeout=180)
    print("/backtest:", r.status_code, r.text)

if __name__ == "__main__":
    main() 