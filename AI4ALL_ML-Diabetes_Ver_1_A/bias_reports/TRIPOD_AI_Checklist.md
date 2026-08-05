# TRIPOD-AI Reporting Checklist

Generated from existing GE-79 project artifacts. This report does not retrain models.

## Source Artifacts Reviewed

- README: `README.md` (found)
- Dataset: `data/GE79_MASTER_DATASET_V1.csv` (found)
- Outputs directory: `outputs` (found)
- NIST AI RMF assessment: `outputs/NIST_AI_RMF_Assessment.md`
- Model result CSVs: Model 1, Model 2, and Model 3 result files.
- Figures reviewed by file presence: confusion matrices, ROC-AUC, PR-AUC, feature importance, and SHAP plots.

## Project Snapshot

- Dataset rows: 75
- Dataset columns: 46
- Target counts: {'Normal': 55, 'Mild Impairment': 20}
- Locked predictor count: 14

## Model Metric Snapshot

| Model | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall |
|---|---:|---:|---:|---:|---:|
| Model 1 - Logistic Regression | 0.547 | 0.525 | 0.532 | 0.508 | 0.5 |
| Model 2 - Decision Tree | 0.627 | 0.583 | 0.602 | 0.58 | 0.55 |
| Model 3 - Random Forest | 0.747 | 0.664 | 0.589 | 0.594 | 0.25 |

## ROC-AUC and PR-AUC Snapshot

| Model | ROC-AUC | PR-AUC | Interpretation |
|---|---:|---:|---|
| Model 1 - Logistic Regression | 0.534 | 0.369 | Weak ranking performance, but stronger impaired recall than Random Forest. |
| Model 2 - Decision Tree | 0.639 | 0.370 | Moderate ROC ranking, but PR-AUC remains limited for the impaired minority class. |
| Model 3 - Random Forest | 0.648 | 0.441 | Best ROC-AUC and PR-AUC among the three, but impaired recall remains low. |

## Top SHAP Features Found

- daytime sbp
- fasting glucose mg dl
- glucose mg dl
- global vasoreactivity
- wmh registered

## Checklist

### Study Title

Status: ✓ Complete

Evidence:
- README contains a project title: GE-79 Cognitive-Status Classification.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Keep the title consistent across README, reports, and presentation materials.

### Research Question

Status: ✓ Complete

Evidence:
- README describes the goal: classify cognitive status from diabetes and cerebrovascular biomarkers.
- README states the class-imbalance problem and the need to evaluate minority-class detection.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- For publication, phrase the objective as a formal prediction-model research question.

### Dataset

Status: ✓ Complete

Evidence:
- Dataset file found: data/GE79_MASTER_DATASET_V1.csv.
- Rows: 75; columns: 46.
- README identifies GE-79 / PhysioNet CDED 1.0.1 as the primary modeling dataset.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Add a data dictionary or direct column-level provenance table if required by reviewers.

### Participants

Status: ⚠ Partial

Evidence:
- Cohort size found from CSV: 75 patients.
- Target counts found from CSV: {'Normal': 55, 'Mild Impairment': 20}.

Gaps:
- Detailed inclusion/exclusion criteria and participant demographics are not fully reported in the generated outputs.

Recommendation:
- Add participant eligibility criteria, recruitment/source cohort details, and demographics if available.

### Outcome Definition

Status: ✓ Complete

Evidence:
- README defines the binary target from MMSE: 0 = No Impairment and 1 = Impaired.
- Model files use `cognitive_status_label` mapped to Normal and Mild Impairment classes.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Document any clinical rationale for the MMSE thresholds in a methods appendix.

### Predictors

Status: ✓ Complete

Evidence:
- Locked feature file found with 14 predictors: outputs/model0_FINAL_FEATURES.csv.
- Predictor domains include glycemic, blood-pressure, vascular, imaging, inflammatory, and anthropometric features.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Include a table mapping each predictor to unit, source column, and clinical domain.

### Data Cleaning

Status: ⚠ Partial

Evidence:
- README reports pandas data ingestion and scikit-learn preprocessing pipelines.
- Existing code uses imputation, scaling, and one-hot encoding where needed.

Gaps:
- A step-by-step raw data cleaning log is not present in the existing outputs.

Recommendation:
- Add a data-cleaning audit table with exclusions, renamed columns, and derived fields.

### Missing Data

Status: ✓ Complete

Evidence:
- README reports median imputation and an explicit missingness indicator for `diabetes_duration`.
- Model pipelines perform imputation inside cross-validation folds.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Add a missingness table by predictor for formal TRIPOD-AI reporting.

### Feature Engineering

Status: ✓ Complete

Evidence:
- The project adds a `diabetes_duration_missing` flag when `diabetes_duration` is selected.
- Categorical features are one-hot encoded inside model pipelines when present.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- List all engineered fields and transformations in a reproducibility appendix.

### Feature Selection

Status: ✓ Complete

Evidence:
- Model 0 selected a locked 14-biomarker feature set.
- Feature importance outputs and feature-selection figures are present.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Report whether any science-based anchor features were forced into the final feature set.

### Machine Learning Models

Status: ✓ Complete

Evidence:
- Existing result CSVs found for Logistic Regression, Decision Tree, and Random Forest.
- README describes Model 1, Model 2, and Model 3 model types.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Add a model-parameter table for all estimators in the final report.

### Validation Strategy

Status: ✓ Complete

Evidence:
- README documents 5-fold stratified cross-validation.
- Model code uses StratifiedKFold and cross_val_predict.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- For future work, add external validation on a separate cohort.

### Performance Metrics

Status: ✓ Complete

Evidence:
- Accuracy, precision macro, recall macro, macro F1, and impaired recall are stored in result CSVs.
- Confusion matrix, ROC-AUC, and PR-AUC figures are present for Models 1-3.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Store ROC-AUC and PR-AUC values in a machine-readable CSV in addition to figures.

### Limitations

Status: ✓ Complete

Evidence:
- NIST assessment found at: outputs/NIST_AI_RMF_Assessment.md.
- Limitations include small cohort size, class imbalance, and lack of external validation.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Add confidence intervals or bootstrap uncertainty estimates for performance metrics.

### Responsible AI

Status: ✓ Complete

Evidence:
- NIST assessment scopes the project as research-only and not for clinical deployment.
- Outputs include plain-language chart explanations and SHAP model explanations.
- VerifyWise fairness screening reported low bias for statistical parity, disparate impact, and demographic parity across tested protected attributes.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Add model-prediction subgroup fairness analysis; VerifyWise screened the dataset target label, not model predictions.

### Bias Review

Status: ✓ Complete

Evidence:
- Bias mitigation file is present.
- README and NIST assessment identify class imbalance and impaired-class recall as central safety concerns.
- VerifyWise did not detect statistical parity differences for race, diabetes status, hypertension status, insulin use, or study group under the selected metrics.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Evaluate subgroup fairness on model predictions, not only ground-truth labels, if subgroup counts are sufficient.

### Reproducibility

Status: ✓ Complete

Evidence:
- requirements.txt is present.
- Source scripts for Models 0-3 are present.
- Outputs are version-controlled as CSV and PNG artifacts.

Gaps:
- No major gap identified from existing outputs.

Recommendation:
- Add a single runbook describing exact execution order for all scripts.

## Overall TRIPOD-AI Completeness Score

- Complete sections: 15
- Partial sections: 2
- Missing sections: 0
- Weighted score: 16.0 / 17.0
- Overall completeness: 94.1%

Scoring rule: Complete = 1.0 point, Partial = 0.5 point, Missing = 0 points.

## Priority Recommendations

- Add participant eligibility criteria, recruitment/source cohort details, and demographics if available.
- Add a data-cleaning audit table with exclusions, renamed columns, and derived fields.

## Notes

- This checklist is based only on files already present in the repository.
- It does not infer undocumented participant criteria, external validation, or clinical readiness.
- It should be reviewed manually before being used in a formal research submission.
