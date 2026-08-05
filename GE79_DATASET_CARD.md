# GE-79 Dataset Card

Generated from existing GE-79 project artifacts. This dataset card documents the cleaned modeling dataset used by the AI4ALL Ignite 2026 GE-79 machine learning project. It does not retrain models and does not introduce new analyses.

## Dataset Summary

- Dataset file: `data/GE79_MASTER_DATASET_V1.csv`
- Source identified in project README: PhysioNet CDED / GE-79, version 1.0.1
- Rows: 75
- Columns: 46
- Target column: `cognitive_status_label`
- Target encoding used by the project: Normal / No Impairment vs Mild Impairment / Impaired
- Project use: research-only cognitive-status classification from clinical and MRI-derived biomarkers

## Data Source

The primary modeling dataset is GE-79 from PhysioNet CDED 1.0.1, as documented in the project README. GE-75 is documented separately as a supporting reference dataset for prospective external validation, but it is not used in the current GE-79 modeling pipeline.

## Participant Population

The project documentation describes the cohort as older adults with Type 2 Diabetes and controls from the GE-79 dataset. The cleaned modeling file contains 75 participants after project curation. All model results are internal to this GE-79 cohort and should not be generalized outside the dataset without external validation.

## Outcome Definition

The modeling target is `cognitive_status_label`, derived from cognitive-status coding used in the project workflow. The classification task distinguishes participants labeled Normal / No Impairment from participants labeled Mild Impairment / Impaired.

## Class Distribution

| Cognitive Status | Count | Percent |
|---|---:|---:|
| Normal | 55 | 73.3% |
| Mild Impairment | 20 | 26.7% |

## Variables

The dataset contains demographic, group-status, clinical, cardiovascular, inflammatory, gait/function, ophthalmologic/vascular, brain-volume, white-matter, vasoreactivity, and cerebral-perfusion variables.

Columns:

`datapatient_id, visit, cognitive_status_code, cognitive_status_label, group, diabetes_duration, htn_status, race, dm_status, height_m, mass_kg, bmi, hba1c_percent, glucose_mg_dl, fasting_glucose_mg_dl, insulin_yes_no, insulin_uiu_ml, daytime_sbp, nighttime_sbp, daytime_dbp, nighttime_dbp, wbc_k_ul, crp_mg_l, sicam_pg_ml, svcam_ng_ml, cholesterol_mg_dl, hdl_mg_dl, ldl_calc_mg_dl, gait_walk1_distance_m, gait_walk1_speed_m_s, gait_dual_task_distance_m, gait_dual_task_speed_m_s, global_gm_vol, global_wm_vol, global_csf_vol, global_intracranial_vol_ml, global_gm_icv, global_wm_icv, global_csf_icv, wmh_registered, wmh_registered_masked, global_vasoreactivity, perfusion_whole_brain_baseline_whole, perfusion_aca_baseline_whole, perfusion_mca_baseline_whole, perfusion_lepto_pca_baseline_whole`

## Cleaning and Preprocessing

The cleaned dataset is used by the project pipeline with documented leakage prevention. MMSE-derived target information is excluded from model predictors. Downstream modeling uses pipeline-based preprocessing, including median imputation, standardization where applicable, and classifier fitting inside cross-validation folds.

## Missing Data

Missingness is present and documented. The project retains `diabetes_duration` with median imputation and an explicit missingness indicator when that feature is used, preserving sample size while making missingness visible to downstream models.

Top missing columns in the cleaned file:

| Column | Missing Count | Missing Percent |
|---|---:|---:|
| gait_walk1_distance_m | 75 | 100.0% |
| diabetes_duration | 34 | 45.3% |
| global_vasoreactivity | 13 | 17.3% |
| nighttime_sbp | 10 | 13.3% |
| nighttime_dbp | 10 | 13.3% |
| daytime_dbp | 9 | 12.0% |
| daytime_sbp | 9 | 12.0% |
| gait_dual_task_distance_m | 7 | 9.3% |
| gait_walk1_speed_m_s | 7 | 9.3% |
| gait_dual_task_speed_m_s | 7 | 9.3% |

## Known Biases and Limitations

- Small cohort size: 75 participants.
- Class imbalance: fewer mild-impairment cases than normal/no-impairment cases.
- Geographic and single-cohort limitations are documented in the risk reports.
- Selection bias and survivorship bias are documented as study-design concerns.
- Limited racial representation is documented as a dataset limitation.
- Label uncertainty remains because the target is based on clinical cognitive-status labeling.
- External validation has not yet been completed.

## Licensing and Access

The project identifies PhysioNet CDED / GE-79 as the data source. Users should follow the applicable PhysioNet dataset access terms, citation expectations, and license information from the original PhysioNet dataset page. This dataset card does not replace the original data-use terms.

## Appropriate Use

- Educational and research analysis within the AI4ALL Ignite project context.
- Internal model comparison across the documented GE-79 workflow.
- Dataset auditing, exploratory analysis, feature-selection review, and Responsible AI documentation.
- Hypothesis generation about biomarkers associated with cognitive-status classification.

## Inappropriate Use

- Clinical diagnosis, triage, treatment decisions, or patient-level medical decision support.
- Claims that the model is externally validated or deployment-ready.
- Claims that the dataset or models are unbiased.
- Generalization to other cohorts without external validation.
- Use of target-derived cognitive-status fields as predictors.

## Responsible AI Notes

This dataset card should be read alongside the NIST AI RMF assessment, TRIPOD-AI checklist, PROBAST-AI report, VerifyWise fairness assessment, OECD AI Principles mapping, and GE-79 model card. Together, these documents support transparent, conservative interpretation of the GE-79 machine learning results.
