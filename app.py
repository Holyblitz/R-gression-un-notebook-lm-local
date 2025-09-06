import os
import pandas as pd
import streamlit as st
from src.predict_ml import predict_for_context_tools_ml

# --- Imports optionnels (si tu as ajouté ces fichiers) ---
try:
    from src.explain import explain_reco
except Exception:
    explain_reco = None
try:
    from src.report import export_markdown
except Exception:
    export_markdown = None

# --- Imports cœur ---/home/romain/port_folio/port_folio_ds/Validation_POC/APP_ML_PATCH_SNIPPET.txt
from src.vrs_extractor import extract_vrs
from src.vrs_score import compute_vrs, vrs_zone, load_config, score_0_2
from src.socratic import next_missing_field, get_question_for_field
from src.ranking import heuristic_pred_days, top_k_by_category

# --- Utilitaire: cibler les champs "faibles" (score qualité < 2) ---
def next_low_field(fields: dict) -> str | None:
    """Retourne le premier champ dont le score qualité < 2 (ordre priorisé)."""
    order = [
        "kpi","kpi_target","data_source","data_format","data_volume",
        "integration_target","sponsor",
        "constraints_latency","constraints_cost","constraints_security",
        "usage_pattern","problem","beneficiary","risks"
    ]
    for k in order:
        val = fields.get(k, "")
        if score_0_2(val) < 2:   # 0=vide, 1=flou/TBD, 2=ok
            return k
    return None

# ---------------------- App ----------------------
st.set_page_config(page_title="Notebook LM (local)", layout="wide")
st.title("Notebook LM (local) · Vision-first ToolFit")

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.markdown("### Modèle LLM")
    model = st.text_input("Model ID (Ollama)", value=os.getenv("LLM_MODEL","mistral:instruct"))
    st.caption("Ex: mistral:instruct, qwen2.5:7b-instruct, etc.")
    st.markdown("---")
    st.markdown("#### Données")
    uploaded_tools = st.file_uploader("tools.csv", type=["csv"], help="Sinon on charge data/tools.csv par défaut")
    if uploaded_tools is not None:
        st.session_state["tools_df"] = pd.read_csv(uploaded_tools)
    else:
        if "tools_df" not in st.session_state:
            try:
                st.session_state["tools_df"] = pd.read_csv("data/tools.csv")
            except Exception:
                st.session_state["tools_df"] = pd.DataFrame()

tabs = st.tabs(["Vision", "Recommandations"])

# ---------------------- Vision ----------------------
with tabs[0]:
    st.subheader("1) Brief → Extraction VRS")
    brief = st.text_area(
        "Colle ton brief POC ici", height=200,
        placeholder="Contexte, objectif, KPI, données, contraintes, sponsor, intégration, risques..."
    )
    colA, colB = st.columns(2)

    # State init
    if "fields" not in st.session_state:
        st.session_state["fields"] = {}
    if "current_field" not in st.session_state:
        st.session_state["current_field"] = None

    # Extraction LLM
    if colA.button("Extraire (LLM local)"):
        st.session_state["fields"] = extract_vrs(brief, model=model)
        st.session_state["current_field"] = next_low_field(st.session_state["fields"])  # démarrer au 1er champ faible

    # Reset
    if colB.button("Réinitialiser"):
        st.session_state["fields"] = {}
        st.session_state["current_field"] = None

    fields = st.session_state["fields"]

    # Affichage VRS
    if fields:
        df_view = pd.DataFrame({"champ": list(fields.keys()), "valeur": list(fields.values())})
        st.dataframe(df_view, use_container_width=True)
        vrs = compute_vrs(fields)
        st.metric("VRS (Vision Readiness Score)", vrs)
        zone = vrs_zone(vrs)
        st.write(f"Zone: **{zone.upper()}**  —  seuils: rouge<60, 60≤orange<75, vert≥75")
        st.progress(vrs/100.0)

        # Détail du VRS (débug)
        with st.expander("Détail du VRS (débug)", expanded=False):
            cfg = load_config()
            rows = []
            for k, w in cfg["weights"].items():
                val = fields.get(k, "")
                s = score_0_2(val)
                rows.append({"champ": k, "valeur": val, "score(0-2)": s, "poids": w, "contrib": s*float(w)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("2) Dialogue socratique (compléter les trous)")

    # Déterminer/maintenir le champ courant (qualité<2)
    if st.session_state["current_field"] is None:
        st.session_state["current_field"] = next_low_field(fields)

    c1, _ = st.columns([1,1])
    with c1:
        if st.button("Question suivante"):
            st.session_state["current_field"] = next_low_field(fields)

    fld = st.session_state.get("current_field")
    if not fld:
        st.success("Tous les champs sont au niveau 'OK'. 👌")
    else:
        # Formulaire : Entrée soumet la réponse
        with st.form(key=f"form_{fld}"):
            st.info(get_question_for_field(fld))
            ans = st.text_input("Ta réponse", value=fields.get(fld, ""))
            submitted = st.form_submit_button("Enregistrer")
        if submitted:
            fields[fld] = ans
            st.session_state["fields"] = fields
            st.session_state["current_field"] = next_low_field(fields)
            st.rerun()

# ---------------------- Recommandations ----------------------
with tabs[1]:
    st.subheader("Ranking d'outils — Time-to-MVP (heuristique v0)")

    tools_df = st.session_state.get("tools_df", pd.DataFrame())
    if tools_df.empty:
        st.warning("Charge `tools.csv` (ou place-le dans data/tools.csv).")
    else:
        st.caption("Renseigne quelques paramètres de contexte (simplifié v0).")
        c1, c2, c3, c4 = st.columns(4)
        lang_fr = c1.selectbox("Langue principale FR ?", ["true","false"], index=0)
        infra = c2.selectbox("Infra dispo", ["cpu","gpu","mixed"], index=0)
        latency = c3.number_input("Latence max (ms)", min_value=10, max_value=5000, value=300, step=10)
        budget = c4.number_input("Budget mensuel (€)", min_value=0, value=500, step=50)

        # VRS courant
        vrs_val = compute_vrs(st.session_state.get("fields", {})) if st.session_state.get("fields") else 0

        context = {"lang_fr": lang_fr, "infra": infra, "latency_ms_budget": latency, "budget_eur": budget}
        scored = heuristic_pred_days(tools_df, context, vrs_val)

        st.write(f"VRS courant: **{vrs_val}**")
        st.dataframe(scored.sort_values(["category","pred_days"]), use_container_width=True, height=400)

        st.markdown("### Top-3 par catégorie")
        topk = top_k_by_category(scored, k=3)
        st.dataframe(topk, use_container_width=True)

        # Explications (optionnel)
        with st.expander("Explications (LLM local)", expanded=False):
            ctx_str = f"lang_fr={lang_fr}, infra={infra}, latency={latency}, budget={budget}, VRS={vrs_val}"
            if explain_reco is None:
                st.info("Module d'explication indisponible (src/explain.py manquant).")
            else:
                if st.button("Expliquer Top-3"):
                    txt = explain_reco(ctx_str, topk, model=model)
                    st.write(txt or "(LLM non disponible)")

        # Export Markdown (optionnel)
        col1, col2 = st.columns(2)
        with col1:
            if export_markdown is None:
                st.caption("Exporter résumé: nécessite src/report.py")
            else:
                if st.button("Exporter résumé Markdown"):
                    try:
                        zone = "red" if vrs_val < 60 else ("orange" if vrs_val < 75 else "green")
                        out = export_markdown(vrs_val, zone, st.session_state.get("fields", {}), topk, out_path="artifacts/report.md")
                        st.success(f"Exporté: {out}")
                    except Exception as e:
                        st.error(f"Échec export: {e}")

        st.caption("Note: si VRS < 60, aucune reco (scores à ∞). En orange, une pénalité est appliquée.")

