"""
GE-79 Interactive Cognitive-Status Prototype — Prediction API
AI4ALL Ignite · Summer Cohort 2026 · Group 6C

Flask backend that:
  - Loads GE-79 data and trains all 3 models on startup (n=75, fast)
  - Serves /api/predict (POST) — returns predictions + probabilities from all 3 models
  - Serves /api/features (GET) — returns feature metadata (ranges, descriptions)
  - Serves the React build from ../client/dist in production
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---- paths ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "GE79_Features_DATASET_V2.csv"
CLIENT_DIST = BASE_DIR / "client" / "dist"

# ---- feature metadata (14 FINAL_FEATURES) --------------------------------
FEATURES = [
    {
        "key": "glucose_mg_dl",
        "label": "Glucose (mg/dL)",
        "domain": "Glycemic",
        "description": "#1 ranked — strongest single predictor; hyperglycemia drives microvascular damage.",
        "unit": "mg/dL",
        "typical_min": 60,
        "typical_max": 200,
        "reference_low": 70,
        "reference_high": 100,
    },
    {
        "key": "fasting_glucose_mg_dl",
        "label": "Fasting Glucose (mg/dL)",
        "domain": "Glycemic",
        "description": "#2 ranked — confirms the glycemic signal under fasting conditions.",
        "unit": "mg/dL",
        "typical_min": 60,
        "typical_max": 200,
        "reference_low": 70,
        "reference_high": 99,
    },
    {
        "key": "hba1c_percent",
        "label": "HbA1c (%)",
        "domain": "Glycemic",
        "description": "Anchor — long-term glycemic control; central to the research question.",
        "unit": "%",
        "typical_min": 4.0,
        "typical_max": 12.0,
        "reference_low": 4.0,
        "reference_high": 5.7,
    },
    {
        "key": "diabetes_duration",
        "label": "Diabetes Duration (years)",
        "domain": "Glycemic",
        "description": "Anchor — longer duration → greater cumulative vascular risk. Often missing; imputed.",
        "unit": "years",
        "typical_min": 0,
        "typical_max": 40,
        "reference_low": 0,
        "reference_high": 10,
        "allow_missing": True,
    },
    {
        "key": "daytime_sbp",
        "label": "Daytime SBP (mmHg)",
        "domain": "Cardiovascular",
        "description": "Top-4 ranked + anchor — systolic BP load; vascular stress on the brain.",
        "unit": "mmHg",
        "typical_min": 90,
        "typical_max": 180,
        "reference_low": 90,
        "reference_high": 130,
    },
    {
        "key": "nighttime_sbp",
        "label": "Nighttime SBP (mmHg)",
        "domain": "Cardiovascular",
        "description": "Ranked — non-dipping nocturnal BP is a known cerebrovascular risk marker.",
        "unit": "mmHg",
        "typical_min": 80,
        "typical_max": 170,
        "reference_low": 80,
        "reference_high": 120,
    },
    {
        "key": "ldl_calc_mg_dl",
        "label": "LDL (mg/dL)",
        "domain": "Cardiovascular",
        "description": "Ranked — lipid burden contributing to vascular disease.",
        "unit": "mg/dL",
        "typical_min": 50,
        "typical_max": 250,
        "reference_low": 50,
        "reference_high": 100,
    },
    {
        "key": "svcam_ng_ml",
        "label": "sVCAM-1 (ng/mL)",
        "domain": "Inflammation",
        "description": "Ranked — endothelial/vascular adhesion marker tied to vasoreactivity decline.",
        "unit": "ng/mL",
        "typical_min": 300,
        "typical_max": 1500,
        "reference_low": 300,
        "reference_high": 900,
    },
    {
        "key": "global_vasoreactivity",
        "label": "Global Vasoreactivity",
        "domain": "Cerebrovascular",
        "description": "Top-3 ranked + anchor — cerebral vasoreactivity; the mechanistic heart of CDED.",
        "unit": "",
        "typical_min": -0.5,
        "typical_max": 3.0,
        "reference_low": 0.3,
        "reference_high": 2.0,
        "allow_missing": True,
    },
    {
        "key": "wmh_registered",
        "label": "WMH Registered (mL)",
        "domain": "Cerebrovascular",
        "description": "Ranked + anchor — white-matter hyperintensities; diabetic small-vessel damage.",
        "unit": "mL",
        "typical_min": 0,
        "typical_max": 50,
        "reference_low": 0,
        "reference_high": 15,
        "allow_missing": True,
    },
    {
        "key": "wmh_registered_masked",
        "label": "WMH Masked (mL)",
        "domain": "Cerebrovascular",
        "description": "Ranked — masked WMH measure; corroborates the white-matter signal.",
        "unit": "mL",
        "typical_min": 0,
        "typical_max": 50,
        "reference_low": 0,
        "reference_high": 15,
        "allow_missing": True,
    },
    {
        "key": "perfusion_whole_brain_baseline_whole",
        "label": "Whole-Brain Perfusion",
        "domain": "Cerebrovascular",
        "description": "Ranked + anchor — whole-brain cerebral perfusion (mL/100g/min).",
        "unit": "mL/100g/min",
        "typical_min": 20,
        "typical_max": 90,
        "reference_low": 40,
        "reference_high": 70,
        "allow_missing": True,
    },
    {
        "key": "perfusion_lepto_pca_baseline_whole",
        "label": "PCA Perfusion",
        "domain": "Cerebrovascular",
        "description": "Ranked — posterior (PCA territory) cerebral perfusion.",
        "unit": "mL/100g/min",
        "typical_min": 20,
        "typical_max": 90,
        "reference_low": 40,
        "reference_high": 70,
        "allow_missing": True,
    },
    {
        "key": "mass_kg",
        "label": "Body Mass (kg)",
        "domain": "Body Composition",
        "description": "Ranked — body mass correlates with metabolic and vascular load.",
        "unit": "kg",
        "typical_min": 40,
        "typical_max": 150,
        "reference_low": 50,
        "reference_high": 90,
    },
]

FEATURE_KEYS = [f["key"] for f in FEATURES]

# ---- feature importance (from Model 3 / Random Forest outputs) -----------
FEATURE_IMPORTANCE = {
    "fasting_glucose_mg_dl": 0.1059,
    "glucose_mg_dl": 0.1036,
    "daytime_sbp": 0.0963,
    "global_vasoreactivity": 0.0959,
    "wmh_registered": 0.0786,
    "svcam_ng_ml": 0.0757,
    "wmh_registered_masked": 0.0683,
    "ldl_calc_mg_dl": 0.0670,
    "perfusion_whole_brain_baseline_whole": 0.0668,
    "nighttime_sbp": 0.0625,
    "mass_kg": 0.0611,
    "perfusion_lepto_pca_baseline_whole": 0.0606,
    "hba1c_percent": 0.0,
    "diabetes_duration": 0.0,
}

# ---- train models on startup ---------------------------------------------
def build_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def train_models():
    df = pd.read_csv(DATA_PATH)
    # drop the synthetic leading column if present
    df.columns = [c.lstrip("data") if c.startswith("datapatient_id") else c for c in df.columns]
    if "datapatient_id" in df.columns:
        df = df.rename(columns={"datapatient_id": "patient_id"})

    y = df["cognitive_status_code"].astype(int)
    X = df[FEATURE_KEYS].copy()

    models = {
        "Logistic Regression": build_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        ),
        "Decision Tree": build_pipeline(
            DecisionTreeClassifier(class_weight="balanced", max_depth=4, random_state=42)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
        ),
    }

    for name, pipe in models.items():
        pipe.fit(X, y)

    return models, X, y


print("Training models on GE-79 data…")
MODELS, X_train, y_train = train_models()
print(f"  Models ready: {list(MODELS.keys())}")

# ---- Flask app -----------------------------------------------------------
app = Flask(__name__, static_folder=str(CLIENT_DIST) if CLIENT_DIST.exists() else None)
CORS(app)


@app.route("/api/features")
def api_features():
    """Return feature metadata and dataset statistics."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.lstrip("data") if c.startswith("datapatient_id") else c for c in df.columns]

    stats = {}
    for key in FEATURE_KEYS:
        col = df[key]
        stats[key] = {
            "mean": round(float(col.mean()), 3),
            "median": round(float(col.median()), 3),
            "min": round(float(col.min()), 3),
            "max": round(float(col.max()), 3),
            "p25": round(float(col.quantile(0.25)), 3),
            "p75": round(float(col.quantile(0.75)), 3),
            "missing_count": int(col.isna().sum()),
        }

    return jsonify({
        "features": FEATURES,
        "stats": stats,
        "feature_importance": FEATURE_IMPORTANCE,
        "n_samples": len(df),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Receive biomarker values, return predictions from all 3 models.
    Body: { "features": { "glucose_mg_dl": 95.0, ... } }
    Sends back None / null for features omitted (will be imputed).
    """
    body = request.get_json(force=True, silent=True)
    if not body or not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    raw = body.get("features", {})
    if not isinstance(raw, dict):
        return jsonify({"error": "'features' must be a JSON object."}), 400

    # Build a single-row DataFrame with the right columns
    row = {}
    validation_errors = []
    for key in FEATURE_KEYS:
        val = raw.get(key)
        if val is None or val == "":
            row[key] = np.nan
        else:
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                validation_errors.append(f"'{key}': expected a number, got {val!r}")
    if validation_errors:
        return jsonify({"error": "Invalid feature values.", "details": validation_errors}), 400

    X_input = pd.DataFrame([row], columns=FEATURE_KEYS)

    results = {}
    for model_name, pipe in MODELS.items():
        pred = int(pipe.predict(X_input)[0])
        proba = pipe.predict_proba(X_input)[0].tolist()
        results[model_name] = {
            "prediction": pred,
            "label": "No Impairment" if pred == 0 else "Impaired",
            "probability_no_impairment": round(proba[0], 4),
            "probability_impaired": round(proba[1], 4),
        }

    return jsonify({
        "predictions": results,
        "input_used": {k: (None if np.isnan(v) else v) for k, v in row.items()},
        "disclaimer": (
            "Research prototype only. Not a diagnostic tool. "
            "GE-79 cohort n=75. Do not use for clinical decisions."
        ),
    })


@app.route("/api/model-metrics")
def api_model_metrics():
    """Return precomputed aggregate metrics for all three models."""
    metrics = [
        {
            "model": "Logistic Regression",
            "accuracy": 0.547,
            "f1_macro": 0.508,
            "recall_impaired": 0.500,
            "roc_auc": 0.534,
            "pr_auc": 0.369,
            "confusion_matrix": [[31, 24], [10, 10]],
            "color": "#4ECDC4",
        },
        {
            "model": "Decision Tree",
            "accuracy": 0.627,
            "f1_macro": 0.580,
            "recall_impaired": 0.550,
            "roc_auc": 0.639,
            "pr_auc": 0.370,
            "confusion_matrix": [[36, 19], [9, 11]],
            "color": "#45B7D1",
        },
        {
            "model": "Random Forest",
            "accuracy": 0.747,
            "f1_macro": 0.594,
            "recall_impaired": 0.250,
            "roc_auc": 0.648,
            "pr_auc": 0.441,
            "confusion_matrix": [[51, 4], [15, 5]],
            "color": "#96CEB4",
        },
    ]
    return jsonify({"models": metrics})


# ---- serve React app in production ---------------------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if CLIENT_DIST.exists():
        target = CLIENT_DIST / path
        if path and target.exists():
            resp = send_from_directory(str(CLIENT_DIST), path)
        else:
            resp = send_from_directory(str(CLIENT_DIST), "index.html")
        # Prevent browser caching so updates are always visible immediately
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    return jsonify({"status": "API running — React build not found. Run: cd client && npm run build"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
