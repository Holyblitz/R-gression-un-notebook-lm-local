from .vrs_score import VRS_FIELDS

QUESTIONS = {
    "problem":        "Quel problème précis veux-tu résoudre, en une phrase courte ?",
    "beneficiary":    "Qui bénéficie directement du POC (fonction/équipe) ?",
    "kpi":            "Quel KPI principal va mesurer le succès (par ex. précision, temps de réponse) ?",
    "kpi_target":     "Quelle cible chiffrée réaliste pour ce KPI (unité + valeur) ?",
    "data_source":    "D’où viennent les données (source système, propriétaire) ?",
    "data_format":    "Quel format de données (texte, PDF, JSON, base SQL, etc.) ?",
    "data_volume":    "Quel volume/ordre de grandeur (ex. 1M lignes, 10k docs) ?",
    "constraints_latency": "Quelle latence maximale tolérée (en ms/seconde) ?",
    "constraints_cost":    "Quel budget mensuel approximatif (hébergement + tokens) ?",
    "constraints_security":"Existe-t-il des contraintes RGPD/IT spécifiques ?",
    "usage_pattern":       "Quel patron d’usage : RAG, extraction structurée, génération JSON, agent ?",
    "sponsor":             "Qui est le sponsor/décideur (nom, rôle) ?",
    "integration_target":  "Où vivra le MVP (outil interne, appli web, CRM, etc.) ?",
    "risks":               "Quels risques majeurs anticipes-tu (données, IT, juridique) ?",
}

def next_missing_field(fields: dict) -> str | None:
    for k in VRS_FIELDS:
        if not fields.get(k, ""):
            return k
    return None

def get_question_for_field(field: str) -> str:
    return QUESTIONS.get(field, f"Complète le champ: {field}")
