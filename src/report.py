import os

def export_markdown(vrs: int, zone: str, fields: dict, topk_df, out_path: str = "artifacts/report.md") -> str:
    """Create a short Markdown report with VRS & Top-3 table."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = ["# Notebook LM — Résumé POC\n",
             f"**VRS**: {vrs} ({zone})\n\n",
             "## Brief structuré\n"]
    for k,v in (fields or {}).items():
        if v:
            lines.append(f"- **{k}**: {v}\n")
    lines.append("\n## Top-3 par catégorie (pred_p50 jours)\n\n")
    cols = [c for c in ["category","tool_id","pred_p50","pred_p90"] if c in topk_df.columns]
    if cols:
        try:
            import pandas as pd
            md = topk_df[cols].to_markdown(index=False)
        except Exception:
            md = topk_df[cols].to_string(index=False)
        lines.append(md + "\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return out_path
