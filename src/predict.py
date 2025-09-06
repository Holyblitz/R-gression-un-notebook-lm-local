
import argparse, os, json
import numpy as np
import pandas as pd

# Local imports (module mode: python -m src.predict)
from .ranking import heuristic_pred_days, top_k_by_category

# Try to import ML helper (optional)
try:
    from .predict_ml import predict_for_context_tools_ml
    HAS_ML = True
except Exception:
    predict_for_context_tools_ml = None
    HAS_ML = False

ART_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def _apply_vrs_rules(df: pd.DataFrame, vrs: float, alpha: float = 0.4) -> pd.DataFrame:
    out = df.copy()
    if vrs < 60:
        # gating
        for col in ("pred_days","pred_p50","pred_p90"):
            if col in out.columns:
                out[col] = np.inf
        return out
    if vrs >= 60 and vrs < 75:
        lvl = pd.to_numeric(out.get("complexity_level", 2), errors="coerce").fillna(2.0)
        pen = alpha * (75 - vrs) * (1 + 0.5*(lvl - 1))
        for col in ("pred_days","pred_p50","pred_p90"):
            if col in out.columns:
                out[col] = out[col] + pen
    return out

def _ensure_keys(row: dict) -> dict:
    # Provide sensible defaults if some context keys are missing
    defaults = {
        "VRS": 70,
        "expected_qps": 1.0,
        "latency_ms_budget": 1500,
        "budget_eur": 500,
        "industry": "generic",
        "data_sensitivity": "mid",
        "lang_fr": "true",
        "hosting": "onprem",
        "team_nlp_level": "mid",
        "infra": "cpu",
    }
    out = dict(defaults)
    out.update({k: v for k, v in row.items() if pd.notna(v)})
    return out

def main():
    ap = argparse.ArgumentParser(description="Score tools for a given POC (p50/p90 if ML artifacts exist, else heuristic).")
    ap.add_argument("--context", required=True, help="CSV contexts path")
    ap.add_argument("--tools", required=True, help="CSV tools path")
    ap.add_argument("--poc_id", required=True, help="POC id to score")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--heuristic", action="store_true", help="Force heuristic instead of ML")
    args = ap.parse_args()

    ctx = pd.read_csv(args.context)
    tls = pd.read_csv(args.tools)

    row = ctx[ctx["poc_id"]==args.poc_id]
    if row.empty:
        raise SystemExit(f"POC {args.poc_id} introuvable dans {args.context}")
    ctx_row = _ensure_keys(row.iloc[0].to_dict())

    # Compute VRS if present, else from default
    vrs_val = float(ctx_row.get("VRS", 0))

    use_ml = (not args.heuristic) and HAS_ML and \
        os.path.exists(os.path.join(ART_DIR, "gbm_p50.joblib")) and \
        os.path.exists(os.path.join(ART_DIR, "gbm_p90.joblib"))

    if use_ml:
        df = predict_for_context_tools_ml(ctx_row, tls)
        df = _apply_vrs_rules(df, vrs_val)
        # Display per-category Top-k by pred_p50
        out = df.sort_values(["category","pred_p50","pred_p90"]).groupby("category", as_index=False).head(args.topk)
        cols = [c for c in ["category","tool_id","pred_p50","pred_p90","VRS","complexity_level"] if c in out.columns]
        print("\nTop-k par catégorie (ML p50/p90):")
        print(out[cols].to_string(index=False))
    else:
        # Heuristic fallback
        context = {
            "lang_fr": str(ctx_row.get("lang_fr","true")),
            "infra": str(ctx_row.get("infra","cpu")),
            "latency_ms_budget": float(ctx_row.get("latency_ms_budget",1500)),
            "budget_eur": float(ctx_row.get("budget_eur",500)),
        }
        scored = heuristic_pred_days(tls, context, vrs_val)
        scored = _apply_vrs_rules(scored.rename(columns={"pred_days":"pred_p50"}), vrs_val)  # rename for consistency
        out = scored.sort_values(["category","pred_p50"]).groupby("category", as_index=False).head(args.topk)
        cols = [c for c in ["category","tool_id","pred_p50","VRS","complexity_level"] if c in out.columns]
        print("\nTop-k par catégorie (heuristique):")
        print(out[cols].to_string(index=False))

if __name__ == "__main__":
    main()
