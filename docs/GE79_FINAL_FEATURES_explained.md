# GE-79 — FINAL_FEATURES (14)
### How Python selected them, and why each domain is represented

**Group 6C · AI4ALL Ignite 2026 · CDED / GE-79**
**Prepared by Elizabeth Hannan · Ver 1.A · 6/25/2026**
**Target encoding:** `0 = No Impairment` (MMSE ≥ 28) · `1 = Impaired` (MMSE 25–27)

---

## How we got here (the selection method)

The features were **not hand-picked**. They were chosen by a reproducible, evidence-based process so the final set is defensible to judges:

1. **Full scope in.** All **41 candidate features** across six biomarker domains (demographics, clinical/diabetes, cardiovascular, labs/inflammation, MRI-structural, cerebral perfusion) were offered to the model — nothing pre-removed except identifiers, the target, a 100%-missing column, and a duplicate column.
2. **Python ranked them.** A **Random Forest** computed an importance score for every feature (how much it helped classify cognitive status correctly).
3. **Stability averaging.** Because the dataset is small (n = 75), a single ranking is noisy. The importance was **averaged over 20 random seeds**, so the ranking reflects signal, not luck.
4. **Science anchors retained.** A short list of biomarkers the CDED literature (Novak et al.) identifies as mechanistically important — glycemic, blood pressure, vasoreactivity, perfusion, white-matter — were **kept regardless of rank**, so the model can't accidentally discard clinically essential variables.
5. **Final set =** top-12 data-ranked features **∪** science anchors = **14 features**, reused by all three models so they are directly comparable.

> **Why this matters for grading:** the same process is repeatable, the ranking is stable, and it ties every feature back to the research question. When narrowed to these 14, even the simple Logistic Regression baseline improved (macro-F1 0.40 → 0.51; impaired-recall 0.25 → 0.45).

---

## The 14 FINAL_FEATURES by domain

### Glycemic / diabetes control
| Feature | Role | Why it's here |
|---|---|---|
| `glucose_mg_dl` | **#1 ranked** | Strongest single predictor; hyperglycemia drives microvascular damage. |
| `fasting_glucose_mg_dl` | **#2 ranked** | Confirms the glycemic signal under fasting conditions. |
| `hba1c_percent` | anchor | Long-term glycemic control; central to the research question. |
| `diabetes_duration` | anchor (kept w/ missing-flag) | Longer duration → greater cumulative vascular risk. 45% missing, so imputed + flagged. |

### Cardiovascular / blood pressure
| Feature | Role | Why it's here |
|---|---|---|
| `daytime_sbp` | **top-4 ranked** + anchor | Systolic BP load; vascular stress on the brain. |
| `nighttime_sbp` | ranked | Non-dipping nocturnal BP is a known cerebrovascular risk marker. |
| `ldl_calc_mg_dl` | ranked | Lipid burden contributing to vascular disease. |

### Inflammation
| Feature | Role | Why it's here |
|---|---|---|
| `svcam_ng_ml` | ranked | sVCAM-1, an endothelial/vascular adhesion marker tied to vasoreactivity decline. |

### Cerebrovascular core (your original topic — validated by the data)
| Feature | Role | Why it's here |
|---|---|---|
| `global_vasoreactivity` | **top-3 ranked** + anchor | Cerebral vasoreactivity — the mechanistic heart of CDED. |
| `wmh_registered` | ranked + anchor | White-matter hyperintensities — diabetic small-vessel damage. |
| `wmh_registered_masked` | ranked | Masked WMH measure; corroborates the white-matter signal. |
| `perfusion_whole_brain_baseline_whole` | ranked + anchor | Whole-brain cerebral perfusion. |
| `perfusion_lepto_pca_baseline_whole` | ranked | Posterior (PCA territory) perfusion. |

### Body composition
| Feature | Role | Why it's here |
|---|---|---|
| `mass_kg` | ranked | Body mass; correlates with metabolic and vascular load. |

---

## Dataset roles (stay accurate)

| Dataset | Role | Used for modeling? |
|---|---|---|
| **GE-79** | **Core** — all three models train + 5-fold cross-validate here | **Yes** |
| **GE-75** | **Separate raw reference** (waveform `.dat`/`.hea`; no matching label/columns) | **No** — documentation only; possible future external validation *after* cleaning |

> GE-75 is **not** a train/test-against set: it is a different cohort with no MMSE-derived 0/1 label and different columns, so a labeled GE-79→GE-75 validation cannot be run without major separate cleaning.

---

## Datasets & links  *(EH NOTES — Private only)*

**[1] Main Dataset — Elderly Diabetes (core modeling set)**
GE-79 · CDED 1.0.1
https://physionet.org/content/cded/1.0.1/

**[2] Supporting Dataset — Diabetes Perfusion / Cerebral (bulk of clinical-data features; raw reference, not merged)**
GE-75 · Cerebral Perfusion in Diabetes 1.0.1
https://physionet.org/content/cerebral-perfusion-diabetes/1.0.1/

---

## The three models (all use these same 14 features)

| # | Model | Label stamped on files/figures | Role |
|---|---|---|---|
| 1 | Logistic Regression | `GE-79 · Logistic Regression · Baseline` | Simple, explainable benchmark |
| 2 | Decision Tree | `GE-79 · Decision Tree · Interpretable` | Visualizes decision rules |
| 3 | Random Forest | `GE-79 · Random Forest · Ensemble` | Strongest expected; feature importance |
