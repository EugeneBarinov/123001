from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import pandas as pd

from AI.inference.predict_service import (
    load_checkpoint,
    predict_from_dataframe,
    preprocess_df,
    load_sklearn,
    predict_sklearn_from_dataframe,
)
from AI.inference.backtest import run_naive_strategy, sharpe_ratio, max_drawdown
from AI.features.indicators import compute_indicators

app = FastAPI(title="Crypto Trading AI API")

CHECKPOINT_PATH = os.environ.get("MODEL_CKPT", "AI/models/checkpoints/model.pt")
SK_MODELS = {
    "svm": os.environ.get("MODEL_SVM", "AI/models/checkpoints/svm_rbf.joblib"),
    "rf": os.environ.get("MODEL_RF", "AI/models/checkpoints/rf.joblib"),
    "logreg": os.environ.get("MODEL_LOGREG", "AI/models/checkpoints/logreg.joblib"),
}
_artifacts_pt = None
_artifacts_sklearn = {}
_LOGS: List[str] = []


def log(msg: str) -> None:
    if len(_LOGS) > 500:
        _LOGS.clear()
    _LOGS.append(msg)


class PredictRequest(BaseModel):
    csv_path: Optional[str] = "AI/data/historical.csv"


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/logs")
async def logs() -> Dict[str, Any]:
    return {"lines": _LOGS[-200:]}


def _ensure_pt():
    global _artifacts_pt
    if _artifacts_pt is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise HTTPException(status_code=400, detail=f"Checkpoint not found at {CHECKPOINT_PATH}. Train model first.")
        _artifacts_pt = load_checkpoint(CHECKPOINT_PATH)
        log(f"Loaded PT model from {CHECKPOINT_PATH}")


def _ensure_sk(model_key: str):
    if model_key not in SK_MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown sklearn model key: {model_key}")
    path = SK_MODELS[model_key]
    if model_key not in _artifacts_sklearn:
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"Sklearn model not found at {path}. Train models first.")
        _artifacts_sklearn[model_key] = load_sklearn(path)
        log(f"Loaded {model_key} from {path}")


@app.post("/predict")
async def predict(req: PredictRequest, model: str = Query("pt", regex="^(pt|svm|rf|logreg)$")) -> Dict[str, Any]:
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.csv_path}")
    df = pd.read_csv(req.csv_path)

    log(f"Predict start model={model} csv={req.csv_path} rows={len(df)}")
    if model == "pt":
        _ensure_pt()
        out = predict_from_dataframe(df, _artifacts_pt)
    else:
        _ensure_sk(model)
        out = predict_sklearn_from_dataframe(df, _artifacts_sklearn[model])
    log(f"Predict done model={model} signal={out['signal']}")
    return out


@app.post("/backtest")
async def backtest(req: PredictRequest, model: str = Query("pt", regex="^(pt|svm|rf|logreg)$")) -> Dict[str, Any]:
    if not os.path.exists(req.csv_path):
        raise HTTPException(status_code=400, detail=f"CSV not found: {req.csv_path}")
    df = pd.read_csv(req.csv_path)

    df_ind = compute_indicators(df)

    if model == "pt":
        _ensure_pt()
        x = preprocess_df(df, _artifacts_pt)
        idx = df_ind.index[-len(x):]

        import torch
        signals = []
        for i in range(len(x)):
            if i + 1 < _artifacts_pt.seq_len:
                signals.append("SELL")
                continue
            window = x[i + 1 - _artifacts_pt.seq_len: i + 1]
            with torch.no_grad():
                xb = torch.from_numpy(window).unsqueeze(0).to(_artifacts_pt.device)
                logits = _artifacts_pt.model(xb)
                prob_up = float(torch.softmax(logits, dim=1).cpu().numpy()[0][1])
            signals.append("BUY" if prob_up >= 0.5 else "SELL")
    else:
        _ensure_sk(model)
        from AI.features.indicators import compute_indicators_extended
        df_ext = compute_indicators_extended(df)
        art = _artifacts_sklearn[model]
        X = art.scaler.transform(df_ext[art.feature_columns].values)
        if hasattr(art.model, "predict_proba"):
            proba = art.model.predict_proba(X)
            prob_up_series = pd.Series([p[1] for p in proba], index=df_ext.index)
        else:
            preds = art.model.predict(X)
            prob_up_series = pd.Series(preds.astype(float), index=df_ext.index)
        idx = prob_up_series.index
        signals = ["BUY" if p >= 0.5 else "SELL" for p in prob_up_series]

    s = pd.Series(signals, index=idx)
    bt = run_naive_strategy(df_ind.loc[idx], s)
    sr = sharpe_ratio(bt["ret"]) if not bt["ret"].empty else 0.0
    mdd = max_drawdown(bt["equity"]) if not bt["equity"].empty else 0.0
    res = {
        "metrics": {"sharpe": sr, "max_drawdown": mdd},
        "last_equity": float(bt["equity"].iloc[-1]) if not bt.empty else 1.0,
    }
    log(f"Backtest done model={model} sharpe={sr:.3f} mdd={mdd:.3f}")
    return res


@app.post("/data/generate")
async def data_generate(rows: int = 1500, out: str = "AI/data/historical.csv") -> Dict[str, Any]:
    from AI.data.generate_synthetic import generate_ohlcv
    import pathlib
    df = generate_ohlcv(n=rows)
    p = pathlib.Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    log(f"Generated synthetic data rows={len(df)} to {out}")
    return {"rows": len(df), "out": out}


@app.post("/train/pt")
async def train_pt(csv_path: str = "AI/data/historical.csv", epochs: int = 1) -> Dict[str, Any]:
    from AI.models.train import TrainConfig, train_loop
    cfg = TrainConfig(csv_path=csv_path, epochs=epochs)
    log(f"Train PT start csv={csv_path} epochs={epochs}")
    train_loop(cfg)
    log("Train PT done")
    return {"status": "ok"}


@app.post("/train/sklearn")
async def train_sklearn(csv_path: str = "AI/data/historical.csv") -> Dict[str, Any]:
    from AI.models.sklearn_models import SkConfig, train_and_save
    cfg = SkConfig(csv_path=csv_path)
    log(f"Train SK start csv={csv_path}")
    train_and_save(cfg)
    log("Train SK done")
    return {"status": "ok"}


@app.post("/news/refresh")
async def news_refresh(channels: Optional[List[str]] = None) -> Dict[str, Any]:
    from AI.news.tg_ingest import Channel, refresh_from_channels
    channels = channels or ["cryptonews"]
    ch = [Channel(source="telegram", channel=c) for c in channels]
    cnt = refresh_from_channels(ch)
    log(f"News refreshed: inserted={cnt}")
    return {"inserted": cnt}
