from .llm_local import chat_ollama

def explain_reco(context_str: str, topk_df, model: str = "mistral:instruct") -> str:
    """Return a 3–5 bullet executive explanation for the Top-3 tools.
    topk_df must have columns: category, tool_id, pred_days (optionally pred_p90).
    """
    cols = [c for c in ["category","tool_id","pred_p50","pred_days","pred_p90"] if c in topk_df.columns]
    lines = ["\t".join(str(x) for x in row) for row in topk_df[cols].to_numpy().tolist()]
    table_txt = "\n".join(lines)

    system = (
        "Tu es un consultant IA. Style direct, concret. \n"
        "Ta sortie: 3 à 5 puces. \n"
        "(1) Pourquoi #1 est pertinent maintenant (contexte + VRS). \n"
        "(2) Quand #2 bat #1 (condition simple). \n"
        "(3) Risque majeur et mitigation. \n"
        "(4) Si utile: prochaine étape (bench/POC) en 1 phrase.\n"
    )
    user = f"CONTEXTE:\n{context_str}\n\nTOPK (cols={cols}):\n{table_txt}\n"
    return chat_ollama(model, system, user, json_mode=False) or "(LLM non disponible)"
