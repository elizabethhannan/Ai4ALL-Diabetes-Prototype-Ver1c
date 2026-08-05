# PROBAST-AI Risk-of-Bias Assessment

Generated from existing GE-79 project artifacts. This report does not retrain models and does not fabricate unavailable information.

## Source Artifacts Reviewed

- README: `README.md` (found)
- Dataset: `data/GE79_MASTER_DATASET_V1.csv` (found)
- NIST AI RMF assessment: `outputs/NIST_AI_RMF_Assessment.md`
- TRIPOD-AI checklist: `TRIPOD_AI_Checklist.md` (found)
- Bias mitigation notes: `outputs/model1_model2_model3_bias_mitigation.txt` (found)
- Model metrics: existing Model 1, Model 2, and Model 3 result CSVs.
- Model figures: existing confusion matrix, ROC-AUC, PR-AUC, feature-importance, and SHAP outputs.

## Project Evidence Snapshot

- Dataset rows: 75
- Dataset columns: 46
- Target distribution: {'Normal': 55, 'Mild Impairment': 20}
- Locked predictor count: 14
- Top SHAP features found: daytime sbp, fasting glucose mg dl, glucose mg dl, global vasoreactivity, wmh registered
- SHAP patient explanation row: 62

## Model Evaluation Snapshot

| Model | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall |
|---|---:|---:|---:|---:|---:|
| Model 1 - Logistic Regression | 0.547 | 0.525 | 0.532 | 0.508 | 0.5 |
| Model 2 - Decision Tree | 0.627 | 0.583 | 0.602 | 0.58 | 0.55 |
| Model 3 - Random Forest | 0.747 | 0.664 | 0.589 | 0.594 | 0.25 |

## Domain-Level PROBAST-AI Judgments

| PROBAST-AI Domain | Risk of Bias | Applicability Concern |
|---|---|---|
| Participants | High | Moderate |
| Predictors | Moderate | Low |
| Outcome | Moderate | Moderate |
| Analysis | Moderate | Moderate |

## Participants

Risk of Bias: High

Applicability Concern: Moderate

Justification:
- Dataset file contains 75 rows.
- Target distribution from dataset: {'Normal': 55, 'Mild Impairment': 20}.
- README describes the cohort as older adults with type-2 diabetes from GE-79 / PhysioNet CDED.
- Bias notes identify geographic, age, selection, survivorship, and racial representation limitations.

Concerns:
- Sample size is small for clinical prediction modeling.
- The impaired class is the minority class.
- Detailed inclusion/exclusion criteria and demographic breakdown are not fully available in generated outputs.
- Bias notes report limited generalizability outside the GE-79 cohort.

Judgment:

High risk of bias because the small, imbalanced, single-cohort sample can produce unstable estimates and limited representation. Applicability is moderate because the population matches the research question, but use outside GE-79 is not established.

## Predictors

Risk of Bias: Moderate

Applicability Concern: Low

Justification:
- Locked feature file contains 14 final predictors.
- README reports biomarker domains including glycemic, cardiovascular, inflammatory, cerebrovascular, and anthropometric predictors.
- Model 0 feature-selection outputs and Model 3 SHAP outputs are present.
- NIST assessment states MMSE-derived target information is excluded from predictors.

Concerns:
- Predictor units and collection timing are not fully tabulated in a machine-readable reporting file.
- Feature importance and SHAP explain model behavior but do not prove clinical causality.
- Feature selection was performed in a small cohort, so rankings may be unstable.

Judgment:

Moderate risk of bias because predictors are documented and target leakage controls are present, but the small cohort and feature-selection instability remain concerns. Applicability is low concern for the intended biomarker prediction question.

## Outcome

Risk of Bias: Moderate

Applicability Concern: Moderate

Justification:
- README defines the binary target from MMSE: 0 = No Impairment and 1 = Impaired.
- Dataset target counts are available and used in reports.
- Bias notes identify label bias as a remaining limitation.

Concerns:
- The outcome is a cognitive-status label derived from cognitive testing, not an independently adjudicated clinical diagnosis in the generated outputs.
- Label bias may remain if cognitive testing misclassifies impairment.
- Clinical rationale for exact MMSE thresholds is not fully expanded in the existing outputs.

Judgment:

Moderate risk of bias because the outcome is explicitly defined and consistently encoded, but label imperfection and threshold rationale are incompletely documented. Applicability is moderate because the endpoint fits the project but should not be treated as a definitive diagnosis.

## Analysis

Risk of Bias: Moderate

Applicability Concern: Moderate

Justification:
- README reports scikit-learn pipelines, imputation inside cross-validation folds, and 5-fold stratified cross-validation.
- Model result CSVs are present for Models 1-3.
- Confusion matrix, ROC-AUC, and PR-AUC figures are present for Models 1-3.
- SHAP global and individual patient explanation outputs are present for Model 3.
- NIST assessment documents leakage prevention and conservative interpretation.

Concerns:
- No external validation output is present.
- No confidence intervals or bootstrap uncertainty estimates are present.
- Class imbalance remains important: Model 3 has high accuracy but low impaired recall.
- Model development and validation are internal to the same small GE-79 dataset.

Judgment:

Moderate risk of bias because leakage controls, stratified cross-validation, and multiple metrics are present, but sample size, imbalance, and absence of external validation prevent a low-risk judgment.

## Applicability

Risk level: Moderate

Justification:
- The project question, biomarkers, and target are aligned with the GE-79 cognitive-status prediction task.
- Applicability outside GE-79 is limited by the single-cohort design, small sample size, and documented representation concerns.
- Existing NIST and bias notes explicitly describe the work as research-only and not clinically deployable.

## Generalizability

Risk level: High

Justification:
- Existing outputs do not document validation on a separate hospital, geography, or cohort.
- Bias mitigation notes identify geographic, age, selection, survivorship, and racial representation limitations.
- The cohort has 75 patients and an imbalanced target distribution, limiting stable generalization claims.

## External Validation

Risk level: High

Justification:
- No external validation output was found.
- The README identifies GE-75 as a supporting reference dataset for prospective external validation, but the current pipeline excludes it from GE-79 modeling.
- All reported model performance is internal to GE-79 cross-validation.

## Clinical Readiness

Risk level: High

Justification:
- Existing outputs state this is a research-only project.
- Model performance shows clinically important tradeoffs; for example, Model 3 has accuracy 0.747 but impaired recall 0.25.
- No prospective validation, calibration assessment, clinical utility analysis, or deployment monitoring plan was found.

## Overall Risk of Bias Summary

Overall Risk of Bias: High

The overall risk is driven by small sample size, class imbalance, incomplete participant detail, lack of external validation, and residual outcome-label uncertainty. The analysis includes important safeguards, including leakage prevention, fold-contained preprocessing, stratified cross-validation, multiple performance metrics, and model explainability outputs. These safeguards improve transparency but do not eliminate the core risk-of-bias concerns.

## Strengths

- Research-only scope is clearly documented.
- Dataset source and target class balance are documented.
- Target leakage controls are described in existing outputs.
- Models use a shared locked feature set from Model 0.
- Preprocessing is performed inside scikit-learn pipelines.
- 5-fold stratified cross-validation is used.
- Accuracy is not reported alone; macro F1, impaired recall, confusion matrices, ROC-AUC, and PR-AUC are included.
- SHAP outputs support Random Forest interpretability.
- VerifyWise fairness screening found no statistical parity differences for tested protected attributes under its selected metrics.
- Bias mitigation notes and NIST AI RMF assessment are present.

## Limitations

- Small cohort size: 75 patients.
- Class imbalance: 55 No Impairment and 20 Impaired participants.
- Participant inclusion/exclusion criteria and demographics are not fully reported in generated outputs.
- External validation was not found.
- Confidence intervals and calibration metrics were not found.
- Outcome labels may contain cognitive-testing label bias.
- VerifyWise evaluated the ground-truth clinical label, not model-generated predictions, so it does not establish model fairness.
- SHAP explains model behavior but does not establish causality.
- Clinical readiness is not established.

## Future Improvements

- Add external validation on an independent cohort before clinical interpretation.
- Add confidence intervals or bootstrap uncertainty estimates for performance metrics.
- Add calibration plots and calibration metrics.
- Add a participant-flow table with inclusion/exclusion criteria.
- Add demographic summaries and subgroup fairness analysis if valid subgroup counts are available.
- Add a missingness table by predictor.
- Add a model-parameter table for all final models.
- Add decision-curve analysis or clinical utility analysis before considering applied use.

## Notes

- This report is generated only from existing repository files.
- Unknown or unavailable information is reported as a limitation rather than inferred.
- This report should be manually reviewed before formal submission.
