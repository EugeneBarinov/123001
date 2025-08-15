import pandas as pd
from typing import List

try:
    import ta
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Package 'ta' is required for indicators. Add it to requirements and install.") from exc


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "close",
    "volume",
    "rsi_14",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_hist",
]


EXTENDED_FEATURE_COLUMNS: List[str] = DEFAULT_FEATURE_COLUMNS + [
    # Bollinger Bands
    "bb_middle",
    "bb_high",
    "bb_low",
    # Stochastic
    "stoch_k",
    "stoch_d",
    # MFI, OBV
    "mfi_14",
    "obv",
    # Williams %R
    "%r_14",
    # Ichimoku (conversion/base)
    "ichimoku_conv",
    "ichimoku_base",
    # Keltner Channels
    "kc_middle",
    "kc_high",
    "kc_low",
    # TRIX
    "trix_15",
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return copy of df with standard TA indicators added.

    Expects columns: ['datetime','open','high','low','close','volume'].
    Adds: RSI(14), EMA(12), EMA(26), MACD, MACD signal, MACD hist.
    Drops initial NaN rows produced by indicators.
    """
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = required.difference(set(df.columns))
        raise ValueError(f"Missing required columns for indicators: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)

    # RSI(14)
    out["rsi_14"] = ta.momentum.RSIIndicator(close=out["close"], window=14).rsi()

    # EMA 12/26
    out["ema_12"] = ta.trend.EMAIndicator(close=out["close"], window=12).ema_indicator()
    out["ema_26"] = ta.trend.EMAIndicator(close=out["close"], window=26).ema_indicator()

    # MACD (fast=12, slow=26, signal=9)
    macd = ta.trend.MACD(close=out["close"], window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    # Remove rows with NaN from indicator warm-up
    out = out.dropna().reset_index(drop=True)
    return out


def compute_indicators_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Compute an extended set of indicators suitable for ML models.

    Includes Bollinger Bands, Stochastic Oscillator, MFI, OBV, Williams %R, Ichimoku (conv/base),
    Keltner Channels, TRIX.
    """
    out = compute_indicators(df)

    bb = ta.volatility.BollingerBands(close=out["close"], window=20, window_dev=2.0)
    out["bb_middle"] = bb.bollinger_mavg()
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(high=out["high"], low=out["low"], close=out["close"], window=14, smooth_window=3)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()

    mfi = ta.volume.MFIIndicator(high=out["high"], low=out["low"], close=out["close"], volume=out["volume"], window=14)
    out["mfi_14"] = mfi.money_flow_index()

    obv = ta.volume.OnBalanceVolumeIndicator(close=out["close"], volume=out["volume"]).on_balance_volume()
    out["obv"] = obv

    wr = ta.momentum.WilliamsRIndicator(high=out["high"], low=out["low"], close=out["close"], lbp=14).williams_r()
    out["%r_14"] = wr

    try:
        ichi = ta.trend.IchimokuIndicator(high=out["high"], low=out["low"], window1=9, window2=26, window3=52)
        out["ichimoku_conv"] = ichi.ichimoku_conversion_line()
        out["ichimoku_base"] = ichi.ichimoku_base_line()
    except Exception:
        out["ichimoku_conv"] = pd.NA
        out["ichimoku_base"] = pd.NA

    try:
        kc = ta.volatility.KeltnerChannel(high=out["high"], low=out["low"], close=out["close"], window=20)
        out["kc_middle"] = kc.keltner_channel_mband()
        out["kc_high"] = kc.keltner_channel_hband()
        out["kc_low"] = kc.keltner_channel_lband()
    except Exception:
        out["kc_middle"] = pd.NA
        out["kc_high"] = pd.NA
        out["kc_low"] = pd.NA

    try:
        trix = ta.trend.TRIXIndicator(close=out["close"], window=15)
        out["trix_15"] = trix.trix()
    except Exception:
        out["trix_15"] = pd.NA

    out = out.dropna().reset_index(drop=True)
    return out


def select_feature_matrix(df_with_indicators: pd.DataFrame, feature_columns: List[str] = None) -> pd.DataFrame:
    """Select and return feature matrix DataFrame with provided or default columns.

    Does not scale values. Use scaler utilities elsewhere.
    """
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    for col in feature_columns:
        if col not in df_with_indicators.columns:
            raise ValueError(f"Feature column '{col}' not found in DataFrame. Available: {list(df_with_indicators.columns)}")
    return df_with_indicators[feature_columns].copy()
