# NIST AI RMF Assessment

GE-79 Cognitive-Status Classification - AI4ALL Ignite 2026 Group 6C

This document summarizes the project using the four core functions of the NIST Artificial Intelligence Risk Management Framework: Govern, Map, Measure, and Manage. The assessment is written for a research-only machine learning project that predicts binary cognitive-status class from diabetes and cerebrovascular biomarkers.

## Project Scope

- Project type: research-only supervised machine learning study.
- Dataset: GE-79 / PhysioNet CDED 1.0.1.
- Prediction target: cognitive status, encoded as `0 = No Impairment` and `1 = Impaired`.
- Cohort size: 75 patients.
- Class balance: 55 No Impairment and 20 Impaired participants.
- Final feature set: 14 locked biomarkers selected by Model 0, with a missingness flag added for `diabetes_duration` when used by downstream models.
- Models evaluated:
  - Model 0: Random Forest Feature Selection.
  - Model 1: Logistic Regression.
  - Model 2: Decision Tree.
  - Model 3: Random Forest.

## Govern

Governance defines the project purpose, accountability, documentation standards, and limits of use.

### Governance Controls

- This is a research-only AI/ML project and is not intended for diagnosis, treatment, triage, or automated clinical decision-making.
- The dataset source is documented in the repository README and project outputs.
- Code is version-controlled in Git/GitHub, supporting reproducibility and transparent review.
- Model outputs are saved in the `outputs/` directory, including metrics, confusion matrices, ROC/PR curves, feature-importance figures, and SHAP explanations.
- The workflow separates feature selection from supervised model evaluation so all downstream models use the same locked feature set.
- The project includes plain-language interpretation notes in visual outputs to support nontechnical review.

### Governance Risks

- The dataset is small, with only 75 participants.
- The impaired class is the minority class, with 20 impaired participants.
- This project may overstate clinical readiness if aggregate accuracy is interpreted without minority-class recall, ROC-AUC, PR-AUC, and confusion matrices.
- The models require additional external validation before any real-world use.

### Governance Position

The project should be presented as a transparent research prototype. It can support learning, hypothesis generation, and model-comparison discussion, but it should not be represented as a validated medical AI system.

## Map

Map identifies the context, intended use, users, data, limitations, and foreseeable risks.

### Intended Use

The intended use is to evaluate whether diabetes-related, cardiovascular, inflammatory, and cerebrovascular biomarkers contain a measurable signal for cognitive-status classification in the GE-79 cohort.

The project is intended to support:

- AI4ALL research presentation.
- Model-comparison education.
- Bias and safety discussion.
- Exploratory biomarker screening.
- Future research planning.

The project is not intended to support:

- Clinical diagnosis.
- Individual patient treatment recommendations.
- Automated screening deployment.
- Replacement of neurocognitive assessment.
- Use outside the documented GE-79 research context.

### Intended Users

- AI4ALL instructors and reviewers.
- Student researchers.
- Technical mentors.
- Nontechnical symposium viewers.
- Future collaborators reviewing the project design.

### Data and Feature Context

Model 0 locked the shared downstream feature set:

- `daytime_sbp`
- `diabetes_duration`
- `fasting_glucose_mg_dl`
- `global_vasoreactivity`
- `glucose_mg_dl`
- `hba1c_percent`
- `ldl_calc_mg_dl`
- `mass_kg`
- `nighttime_sbp`
- `perfusion_lepto_pca_baseline_whole`
- `perfusion_whole_brain_baseline_whole`
- `svcam_ng_ml`
- `wmh_registered`
- `wmh_registered_masked`

These features represent glycemic, blood-pressure, vascular, imaging, inflammatory, and anthropometric domains.

### Dataset Limitations

- Small cohort size limits statistical certainty.
- The impaired class is underrepresented relative to the no-impairment class.
- The target is derived from cognitive-status labeling and should not be interpreted as a complete clinical diagnosis.
- Missing data exist, especially for `diabetes_duration`, which requires imputation and a missingness flag.
- The analysis is internal to GE-79 and does not establish generalization to other cohorts.

### Mapped Risks

- Majority-class bias: models may perform well on accuracy while missing impaired participants.
- False reassurance risk: false negatives may incorrectly classify impaired participants as no impairment.
- False alarm risk: false positives may classify no-impairment participants as impaired.
- Overfitting risk: small sample size makes model variance more likely.
- Interpretability risk: feature importance and SHAP values explain model behavior, not causal biology.

## Measure

Measure evaluates model performance, uncertainty, bias-relevant behavior, and interpretability outputs.

### Evaluation Design

- 5-fold Stratified Cross-Validation preserves class balance across folds.
- Preprocessing is performed inside scikit-learn pipelines to reduce leakage risk.
- Metrics include accuracy, precision, recall, macro F1-score, confusion matrix, ROC-AUC, and PR-AUC.
- PR-AUC is included because the impaired class is smaller and minority-class performance is safety-relevant.

### Model Performance Summary

| Model | ML Type | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall |
|---|---|---:|---:|---:|---:|---:|
| Model 1 | Logistic Regression | 0.547 | 0.525 | 0.532 | 0.508 | 0.500 |
| Model 2 | Decision Tree | 0.627 | 0.583 | 0.602 | 0.580 | 0.550 |
| Model 3 | Random Forest | 0.747 | 0.664 | 0.589 | 0.594 | 0.250 |

### ROC-AUC and PR-AUC

The project added ROC and Precision-Recall evaluation using predicted probabilities.

| Model | ROC-AUC | PR-AUC | Interpretation |
|---|---:|---:|---|
| Model 1 - Logistic Regression | 0.534 | 0.369 | Weak ranking performance, but stronger impaired recall than Random Forest. |
| Model 2 - Decision Tree | 0.639 | 0.370 | Moderate ROC ranking, but PR-AUC remains limited for the impaired minority class. |
| Model 3 - Random Forest | 0.648 | 0.441 | Best ROC-AUC and PR-AUC among the three, but impaired recall remains low. |

### Bias Review

- Accuracy alone is not sufficient because the majority-class baseline accuracy is 0.733.
- Model 3 accuracy is 0.747, but impaired recall is only 0.250, meaning it misses many impaired cases.
- Model 1 has lower accuracy at 0.547 but catches more impaired participants, with impaired recall of 0.500.
- Model 2 has the highest impaired recall at 0.550 in the current outputs.
- The safest model choice depends on the project goal. If the goal is screening sensitivity, impaired recall and PR-AUC should be weighted more heavily than accuracy.

### VerifyWise Fairness Assessment

The cleaned GE-79 dataset was evaluated using the VerifyWise AI Bias Detector to assess statistical fairness across multiple protected attributes, including race, diabetes status, hypertension status, insulin use, and study group. VerifyWise reported low bias across the tested analyses, with statistical parity difference of 0.000, disparate impact ratio of 1.000, and demographic parity ratio of 1.000.

These results should be interpreted narrowly. They indicate that VerifyWise did not detect measurable parity differences under the selected metrics. They do not prove that the dataset is unbiased, that the trained models are unbiased, or that any individual model is fair. VerifyWise evaluated `cognitive_status_label`, a ground-truth clinical label, rather than model-generated predictions or deployment decisions. Model fairness would require subgroup evaluation of model predictions.

Known study-design limitations remain present despite the VerifyWise results: geographic bias, selection bias, survivorship bias, small sample size, limited racial representation, label uncertainty, and lack of external validation. VerifyWise is therefore treated as one component of a broader Responsible AI assessment alongside documented dataset auditing, bias review, SHAP interpretability, PROBAST-AI review, TRIPOD-AI reporting, and governance practices informed by the NIST AI Risk Management Framework and OECD AI Principles.

### Interpretability Measures

Model 3 includes SHAP explainability outputs:

- `model3_shap_global_importance.csv`
- `model3_shap_patient_explanation.csv`
- `model3_fig_shap_global_importance.png`
- `model3_fig_shap_summary.png`
- `model3_fig_shap_beeswarm.png`
- `model3_fig_shap_waterfall.png`

Top SHAP global features by mean absolute impact include:

- Daytime systolic blood pressure.
- Fasting glucose.
- Glucose.
- Global vasoreactivity.
- White matter hyperintensity measures.

SHAP outputs support model transparency by showing which inputs most influence Random Forest predictions. They do not prove causation and should be interpreted as model-behavior explanations only.

## Manage

Manage identifies risk controls, mitigations, monitoring needs, and conservative interpretation requirements.

### Leakage Prevention

- MMSE-derived target information is not used as a predictor.
- Identifier columns and target-copy columns are excluded from modeling.
- Preprocessing steps are placed inside scikit-learn pipelines.
- Imputation and scaling are fit within cross-validation folds rather than on the full dataset before validation.
- All supervised models use the same locked Model 0 feature set for fair comparison.

### Missing-Data Handling

- Numeric features are median-imputed.
- Categorical features, when present, are mode-imputed and one-hot encoded.
- `diabetes_duration` missingness is explicitly represented with a missingness flag when that feature is used.
- Missing-data handling preserves sample size while making missingness visible to the model.

### Cross-Validation and Reproducibility

- 5-fold stratified cross-validation is used for supervised evaluation.
- Random seeds are fixed to support reproducibility.
- Outputs are saved as CSV and PNG files in `outputs/`.
- Git version control preserves changes to code, model outputs, and documentation.

### Conservative Interpretation

- The models should be described as exploratory research tools.
- Results should be framed around class imbalance and minority-class safety.
- Model 3 should not be selected based only on accuracy because impaired recall is low.
- Model 2 and Model 1 should be discussed for their stronger impaired recall, even if aggregate accuracy is lower.
- SHAP explanations should be used to explain model behavior, not biological causality.

### Risk Management Actions

- Report macro F1, impaired recall, ROC-AUC, and PR-AUC alongside accuracy.
- Include confusion matrices for every supervised model.
- Use ROC and PR curves to communicate threshold tradeoffs.
- Use SHAP to support transparency for Model 3.
- Avoid deployment claims.
- Recommend external validation before any clinical interpretation.
- Recommend review by clinical and ethics experts before future applied use.

## Overall NIST AI RMF Summary

This project demonstrates a responsible research workflow for small-cohort biomedical AI:

- Govern: the project is documented, version-controlled, and scoped as research-only.
- Map: intended use, users, dataset limitations, and foreseeable risks are identified.
- Measure: performance is evaluated with imbalance-aware metrics, ROC/PR analysis, confusion matrices, and SHAP explainability.
- Manage: leakage controls, missing-data handling, stratified cross-validation, and conservative interpretation reduce but do not eliminate risk.

The primary safety finding is that accuracy can be misleading in this dataset. The impaired class is smaller and clinically important, so recall, macro F1, PR-AUC, and confusion matrices are essential for honest model evaluation.
