# GE-79 Interactive Cognitive-Status Prototype

**AI4ALL Ignite · Summer Cohort 2026 · Group 6C**  
Elizabeth Hannan · Agastyya Kola

## Project overview

An interactive, judge-facing research prototype based on the GE-79 diabetes and cognitive-status ML study. Users adjust 14 biomarker sliders and receive real-time predictions from three classifiers (Logistic Regression, Decision Tree, Random Forest) with probability bars, a radar chart, and an aggregate model metrics dashboard.

**This is a research proof-of-concept, not a diagnostic or clinical tool.**

## Stack

| Layer | Technology |
|---|---|
| Frontend | React 18 + TypeScript + Vite, D3 v7 |
| Backend | Python Flask (serves React build + `/api/*` endpoints) |
| ML | scikit-learn — models trained on startup from `data/GE79_Features_DATASET_V2.csv` |
| Dataset | GE-79 / CDED 1.0.1 · n=75 · 5-fold stratified CV |

## How to run

The app runs as a single process — Flask trains the models on startup (fast, n=75) and serves both the API and the pre-built React frontend.

```bash
# 1. Build the React frontend (only needed after frontend changes)
cd client && npm run build && cd ..

# 2. Start the server (configured as the default Replit workflow)
python server/app.py
```

The default Replit workflow (`Start application`) runs `python server/app.py` on port 5000.

## Directory structure

```
server/
  app.py              Flask backend — model training + prediction API + static file serving
client/
  src/
    App.tsx           Root component (tabs: Predict / Model Metrics)
    components/
      BiomarkerForm.tsx    14-feature input panel with sliders + domain grouping
      RadarChart.tsx       D3 radar chart — user profile vs. cohort median (top 8 features)
      PredictionPanel.tsx  Consensus + per-model probability bars
      ModelMetricsPanel.tsx D3 bar chart + confusion matrices + table
  dist/               Production build (served by Flask)
data/
  GE79_Features_DATASET_V2.csv   14-feature dataset used for training
  GE79_MASTER_DATASET_V1.csv     Original master dataset
outputs/              Pre-computed model figures, metrics CSVs, SHAP outputs
src/                  Original model-training scripts (Model 0–3)
app/                  Streamlit research dashboards (deployed on Streamlit Cloud)
docs/                 Feature documentation
```

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/features` | GET | Feature metadata, cohort stats, feature importance |
| `/api/predict` | POST | Predictions from all 3 models for a given biomarker input |
| `/api/model-metrics` | GET | Pre-computed aggregate metrics for all 3 models |

## After frontend changes

```bash
cd client && npm run build
# Then restart the Replit workflow
```

## Clinical disclaimer

This student prototype is for research demonstration, education, and competition purposes within AI4ALL Summer Cohort 2026. It is not intended to diagnose, screen, or treat any condition. Outputs must not replace evaluation or advice from a qualified healthcare professional. The model was developed from n=75 participants and has not been externally validated.

## Dataset citation

Novak, V., & Quispe, R. (2022). *Cerebromicrovascular disease in elderly with diabetes* (Version 1.0.1). PhysioNet. https://doi.org/10.13026/00bm-0x81

## User preferences

- Keep research disclaimers visible and accurate — this is a student competition prototype
- Maintain the existing directory structure (src/, app/, data/, outputs/) alongside the new prototype code
- Use dark theme (--bg: #0a0f1e) matching the project's neumorphic visualization aesthetic
