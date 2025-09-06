import numpy as np
import pandas as pd

def heuristic_pred_days(tools: pd.DataFrame, context: dict, VRS: int, alpha=0.4) -> pd.DataFrame:
    # Base estimate per tool
    out = tools.copy()
    out["pred_days"] = 10.0  # base
    
    # Complexity
    lvl = out.get("complexity_level")
    if lvl is not None:
        out["pred_days"] += (out["complexity_level"].fillna(2).astype(float) - 1) * 3.0
    
    # Hosting penalties
    if str(context.get("infra","cpu")).lower() == "cpu":
        out["pred_days"] += np.where(out["self_host"].fillna(False) & (out["req_gpu_mem_gb"].fillna(0) > 8), 5.0, 0.0)
    
    # Latency mismatch
    lat_budget = float(context.get("latency_ms_budget", 300) or 300)
    out["pred_days"] += np.where(out["avg_latency_ms"].fillna(lat_budget) > lat_budget, 2.5, 0.0)
    
    # FR quality bonus
    lang_fr = str(context.get("lang_fr","true")).lower() in ("true","1","yes","oui")
    if lang_fr:
        out["pred_days"] -= np.where(out["fr_quality"].fillna("mid").astype(str).str.lower()=="high", 1.5, 0.0)
    
    # API availability bonus
    out["pred_days"] -= np.where(out["api_available"].fillna(True), 1.0, 0.0)
    
    # VRS penalty & gating
    if VRS < 60:
        out["pred_days"] = np.inf
    elif VRS < 75:
        level = out["complexity_level"].fillna(2).astype(float)
        out["pred_days"] += alpha * (75 - VRS) * (1 + 0.5*(level - 1))
    
    return out

def top_k_by_category(df: pd.DataFrame, k=3):
    return (df.sort_values(["category","pred_days"])
              .groupby("category", as_index=False)
              .head(k))
