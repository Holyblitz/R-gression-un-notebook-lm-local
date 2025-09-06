# Regression-un-'notebooklm'-local

# Notebook LM (local) · Vision-first ToolFit

Un mini **Notebook LM local-first** (Streamlit + Ollama) qui :

- extrait ton brief et calcule un **VRS – Vision Readiness Score**,
- te pose 2–3 **questions socratiques** pour clarifier,
- **classe les outils** (LLM, vectordb, orchestrateur, guardrails, eval) par **Time-to-MVP**,
- option **ML** : affiche **pred_p50/pred_p90** (jours) via modèles quantiles (sklearn).

> Si **VRS < 60** → gating (∞) ; **60–74** → pénalité ; **≥ 75** → GO.

---

## 👀 Demo

Ajoute 2 captures dans `assets/` et référence-les ici :

- `assets/vrs_100.png` → VRS = 100
- `assets/top3_ml.png` → Top‑3 par catégorie (ML p50/p90)

![VRS 100](assets/vrs_100.png)
![Top-3 ML](assets/top3_ml.png)

---

## 🚀 Quickstart

```bash
# 1) Créer l'environnement
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Installer
pip install -r requirements.txt

# 3) Arbo minimale
mkdir -p data artifacts config src assets

# 4) Config VRS
# thresholds: red<60, 60<=orange<75, green>=75
# alpha_penalty: 0.4 (pénalité zone orange)
# (ex de fichier)
cat > config/weights.yaml << 'YAML'
weights:
  kpi: 2.0
  kpi_target: 2.0
  data_source: 2.0
  data_format: 1.0
  data_volume: 1.0
  integration_target: 1.5
  sponsor: 1.5
  problem: 1.0
  beneficiary: 1.0
  constraints_latency: 1.0
  constraints_cost: 1.0
  constraints_security: 1.0
  usage_pattern: 1.0
  risks: 1.0
thresholds: { red: 60, orange: 75, green: 100 }
alpha_penalty: 0.4
YAML

# 5) Données
# Option A : utiliser les CSV d'exemple (voir plus bas)
# Option B : charger vos CSV via l’uploader Streamlit

# 6) Lancer l’app
streamlit run app.py
```

---

## 📦 Données d’exemple

Place ces fichiers dans `data/` (ou charge-les via la sidebar) :

- `data/tools.csv` (starter inclus)
- `data/contexts.csv` (synthétique)
- `data/historical_pocs.csv` (synthétique)

👉 Fichiers fournis dans ce dépôt (ou à télécharger depuis la release) :

- `data/tools.csv` (starter)
- `data/contexts.csv` (exemple)
- `data/historical_pocs.csv` (exemple)

**Schémas rapides**

`tools.csv`

| col                                                          | type      | notes                                         |
| ------------------------------------------------------------ | --------- | --------------------------------------------- |
| tool_id                                                      | str       | identifiant unique                            |
| category                                                     | str       | llm, vectordb, orchestrator, guardrails, eval |
| open_source, api_available, self_host, json_mode, function_calling | int (0/1) | booléens                                      |
| max_context, tok_cost_per_1k, avg_latency_ms, req_gpu_mem_gb | num       | estimations ok                                |
| fr_quality                                                   | str       | low/mid/high                                  |
| index_type                                                   | str       | (vectordb) ex. ivf_flat, hnsw                 |
| complexity_level                                             | int       | 1=low, 2=mid, 3=high                          |

`contexts.csv`
| poc_id | VRS | latency_ms_budget | budget_eur | infra | lang_fr | … |

`historical_pocs.csv`
| poc_id | tool_id | time_to_mvp_days |

---

## 🧠 Mode ML (p50/p90)

Les modèles (sklearn GradientBoosting, quantiles 0.5 & 0.9) sont chargés depuis `artifacts/` :

```
artifacts/
├─ gbm_p50.joblib
├─ gbm_p90.joblib
└─ feature_meta.json
```

> Pour entraîner vos propres modèles, utilisez le script `notebooks/` ou le générateur synthétique fourni (voir release / docs).

Dans l’onglet **Recommandations**, coche “Utiliser le modèle ML (p50/p90)” ou utilisez la CLI :

```bash
python -m src.predict --context data/contexts.csv --tools data/tools.csv --poc_id POC001 --topk 3
```

---

## 🗂️ Arborescence

```
.
├─ app.py
├─ src/
│  ├─ vrs_extractor.py      # extraction champs via LLM local (Ollama)
│  ├─ vrs_score.py          # calcul VRS + règles
│  ├─ socratic.py           # questions guidées
│  ├─ ranking.py            # heuristique Time-to-MVP
│  ├─ predict_ml.py         # inférence p50/p90
│  └─ predict.py            # CLI scoring
├─ data/
│  ├─ tools.csv
│  ├─ contexts.csv
│  └─ historical_pocs.csv
├─ artifacts/
│  ├─ gbm_p50.joblib
│  ├─ gbm_p90.joblib
│  └─ feature_meta.json
├─ config/
│  └─ weights.yaml
└─ assets/
   ├─ vrs_100.png
   └─ top3_ml.png
```

---

## 🔧 Variables utiles

- `LLM_MODEL` (env) : modèle Ollama (par défaut `mistral:instruct`)
- `OLLAMA_URL` (env) : `http://localhost:11434`

---

## 📜 Licence

MIT — voir `LICENSE`.

---

## 🗺️ Roadmap

- [ ] Feature importance + SHAP locales (explicabilité ML)  
- [ ] Bench latence local (remplir auto `avg_latency_ms`)  
- [ ] Connecteurs (Git, Confluence) pour ingestion doc  
- [ ] Export PDF du rapport

---

## 🙌 Crédits

Conçu pour aider à **sécuriser la vision** avant la stack. Local-first, FR-ready.
