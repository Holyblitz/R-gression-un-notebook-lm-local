import yaml, os

VRS_FIELDS = [
    "problem","beneficiary","kpi","kpi_target","data_source","data_format","data_volume",
    "constraints_latency","constraints_cost","constraints_security",
    "usage_pattern","sponsor","integration_target","risks"
]

def load_config(path:str = None):
    path = path or os.path.join(os.path.dirname(__file__), "..", "config", "weights.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def score_0_2(val: str) -> int:
    if not isinstance(val, str) or not val.strip():
        return 0
    lower = val.lower()
    if any(x in lower for x in ["tbd", "à définir", "a definir", "unknown", "?", "n/a"]):
        return 1
    return 2

def compute_vrs(fields: dict, cfg=None) -> int:
    cfg = cfg or load_config()
    weights = cfg["weights"]
    pts = 0.0
    max_pts = 0.0
    for k, w in weights.items():
        val = fields.get(k, "") if k in fields else ""
        s = score_0_2(val)
        pts += s * float(w)
        max_pts += 2 * float(w)
    vrs = round(100 * pts / max_pts) if max_pts > 0 else 0
    return vrs

def vrs_zone(vrs: int, cfg=None) -> str:
    cfg = cfg or load_config()
    thresh = cfg["thresholds"]
    if vrs < thresh["red"]:
        return "red"
    if vrs < thresh["orange"]:
        return "orange"
    return "green"
