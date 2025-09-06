# src/predict_ml.py
import os
import json
import numpy as np
import pandas as pd
from joblib import load

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def _load_models():
    p50 = load(os.path.join(ART_DIR, "gbm_p50.joblib"))
    p90 = load(os.path.join(ART_DIR, "gbm_p90.joblib"))
    with open(os.path.join(ART_DIR, "feature_meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return p50, p90, meta

def predict_for_context_tools_ml(context_row, tools_df):
    """
    Return a DataFrame with pred_p50 and pred_p90.
    Expects artifacts in ../artifacts/: gbm_p50.joblib, gbm_p90.joblib, feature_meta.json
    """
    p50, p90, meta = _load_models()
    NUM = meta.get("NUM", [])
    CAT = meta.get("CAT", [])

    # Cartesian product contexte × outils
    ctx_df = pd.DataFrame([context_row])
    df = tools_df.copy()
    df["_k"] = 1
    ctx_df["_k"] = 1
    df = df.merge(ctx_df, on="_k").drop(columns="_k")

    # Numériques → float, NaN → 0.0
    for c in NUM:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Catégorielles → str, NaN → ""
    for c in CAT:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype(str).fillna("")

    X_num = df[NUM] if NUM else pd.DataFrame(index=df.index)
    X_cat = df[CAT] if CAT else pd.DataFrame(index=df.index)
    X = pd.concat([X_num, X_cat], axis=1)

    pred_p50 = p50.predict(X)
    pred_p90 = p90.predict(X)
    df["pred_p50"] = np.maximum(pred_p50, 0.0)
    df["pred_p90"] = np.maximum(pred_p90, df["pred_p50"])
    return df

