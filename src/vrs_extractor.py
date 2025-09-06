import json
from .llm_local import chat_ollama

VRS_FIELDS = [
    "problem","beneficiary","kpi","kpi_target","data_source","data_format","data_volume",
    "constraints_latency","constraints_cost","constraints_security",
    "usage_pattern","sponsor","integration_target","risks"
]

VRS_SYSTEM = (
 "Tu es un assistant d’ingénierie POC. "
 "Analyse un brief et RENDS STRICTEMENT un JSON à plat avec ces clés: "
 + ",".join(VRS_FIELDS) +
 ". Si une info manque, mets \"\". Réponds seulement en JSON, sans texte autour."
)

def _try_parse_json(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # tente d'extraire le premier bloc {...}
        if "{" in s and "}" in s:
            s2 = s[s.find("{"): s.rfind("}")+1]
            try:
                return json.loads(s2)
            except Exception:
                return None
    return None

def extract_vrs(brief: str, model: str = "mistral:instruct") -> dict:
    if not brief or not brief.strip():
        return {k: "" for k in VRS_FIELDS}
    # 1er essai
    raw = chat_ollama(model, VRS_SYSTEM, f"BRIEF:\n{brief}", json_mode=True)
    data = _try_parse_json(raw) or {}
    # 2e essai si vide: renforce la contrainte
    if not data:
        skeleton = "{" + ",".join([f'\"{k}\":\"\"' for k in VRS_FIELDS]) + "}"
        hard_system = VRS_SYSTEM + " RENDS EXACTEMENT ce JSON EN REMPLISSANT les valeurs: " + skeleton
        raw2 = chat_ollama(model, hard_system, f"BRIEF:\n{brief}", json_mode=True)
        data = _try_parse_json(raw2) or {}
    return {k: data.get(k, "") for k in VRS_FIELDS}
