import importlib
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mods = [
    'AI.features.indicators',
    'AI.models.model',
    'AI.models.train',
    'AI.inference.predict_service',
    'AI.inference.backtest',
    'AI.api.app',
]
status = {}
for m in mods:
    try:
        importlib.import_module(m)
        status[m] = 'ok'
    except Exception as e:
        status[m] = f'ERR: {e}'
print(status) 