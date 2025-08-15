from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import joblib

from AI.features.indicators import compute_indicators, select_feature_matrix, compute_indicators_extended
from AI.models.model import build_model


@dataclass
class InferenceArtifacts:
    model: torch.nn.Module
    device: torch.device
    feature_columns: List[str]
    seq_len: int
    input_size: int
    mean: np.ndarray
    std: np.ndarray


@dataclass
class SklearnArtifacts:
    model: Any
    scaler: Any
    feature_columns: List[str]


def load_checkpoint(path: str) -> InferenceArtifacts:
    ckpt = torch.load(path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        use_attention=ckpt["use_attention"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return InferenceArtifacts(
        model=model,
        device=device,
        feature_columns=ckpt["feature_columns"],
        seq_len=int(ckpt["seq_len"]),
        input_size=int(ckpt["input_size"]),
        mean=np.asarray(ckpt["standardize_mean"], dtype=np.float32),
        std=np.asarray(ckpt["standardize_std"], dtype=np.float32),
    )


def load_sklearn(path: str) -> SklearnArtifacts:
    bundle = joblib.load(path)
    return SklearnArtifacts(
        model=bundle["model"],
        scaler=bundle["scaler"],
        feature_columns=bundle["feature_columns"],
    )


def preprocess_df(df: pd.DataFrame, artifacts: InferenceArtifacts) -> np.ndarray:
    df_ind = compute_indicators(df)
    feat_df = select_feature_matrix(df_ind, artifacts.feature_columns)
    x = (feat_df.values - artifacts.mean) / np.where(artifacts.std == 0, 1.0, artifacts.std)
    return x.astype(np.float32)


def predict_from_dataframe(df: pd.DataFrame, artifacts: InferenceArtifacts) -> Dict[str, Any]:
    x = preprocess_df(df, artifacts)
    if len(x) < artifacts.seq_len:
        raise ValueError(f"Not enough rows for sequence length {artifacts.seq_len}. Got {len(x)}")
    window = x[-artifacts.seq_len:]
    xb = torch.from_numpy(window).unsqueeze(0).to(artifacts.device)  # (1, T, F)
    with torch.no_grad():
        logits = artifacts.model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {
        "probs": {"down": float(probs[0]), "up": float(probs[1])},
        "signal": "BUY" if probs[1] >= 0.5 else "SELL",
    }


def predict_sklearn_from_dataframe(df: pd.DataFrame, artifacts: SklearnArtifacts) -> Dict[str, Any]:
    df_ind = compute_indicators_extended(df)
    X_df = df_ind[artifacts.feature_columns]
    X = artifacts.scaler.transform(X_df.values)
    proba = None
    if hasattr(artifacts.model, "predict_proba"):
        proba = artifacts.model.predict_proba(X)[-1]
    pred = artifacts.model.predict(X)[-1]
    if proba is not None and len(proba) == 2:
        return {"probs": {"down": float(proba[0]), "up": float(proba[1])}, "signal": "BUY" if int(pred) == 1 else "SELL"}
    else:
        # Fallback: pseudo-probs
        return {"probs": {"down": 1.0 - float(pred), "up": float(pred)}, "signal": "BUY" if int(pred) == 1 else "SELL"}


def predict_from_csv(csv_path: str, artifacts: InferenceArtifacts) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    return predict_from_dataframe(df, artifacts)
