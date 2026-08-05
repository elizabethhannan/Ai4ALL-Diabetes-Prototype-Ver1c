# GE-79 Model Card

Generated from existing GE-79 project artifacts. This model card does not retrain models and does not introduce new analyses.

## Model Card Summary

- Project: GE-79 Cognitive-Status Classification.
- Task: binary classification of cognitive status from diabetes and cerebrovascular biomarkers.
- Target encoding: `0 = No Impairment`; `1 = Impaired`.
- Project status: research-only educational machine learning project.
- Clinical status: not validated for diagnosis, treatment, triage, or deployment.

## Source Artifacts Used

- README: `README.md` (found)
- Dataset: `data/GE79_MASTER_DATASET_V1.csv` (found)
- Model 0 features: `outputs/model0_FINAL_FEATURES.csv` (found)
- Model 1 metrics: `outputs/model1_results_model1_logreg.csv` (found)
- Model 2 metrics: `outputs/model2_results_tree.csv` (found)
- Model 3 metrics: `outputs/model3_results_model3_forest.csv` (found)
- NIST AI RMF assessment: `outputs/NIST_AI_RMF_Assessment.md`
- TRIPOD-AI checklist: `TRIPOD_AI_Checklist.md` (found)
- PROBAST-AI report: `PROBAST_AI_Report.md` (found)
- Bias mitigation notes: `outputs/model1_model2_model3_bias_mitigation.txt` (found)

## Intended Use

The intended use is research and education: to evaluate whether diabetes-related, cardiovascular, inflammatory, cerebrovascular, and anthropometric biomarkers contain predictive signal for cognitive-status classification in the GE-79 cohort.

Appropriate uses:

- AI4ALL symposium presentation.
- Model comparison and responsible AI discussion.
- Exploratory biomarker-screening research.
- Portfolio documentation for a healthcare ML workflow.

Not appropriate uses:

- Clinical diagnosis.
- Treatment decisions.
- Patient triage.
- Automated screening deployment.
- Generalization outside GE-79 without external validation.

## Dataset

- Dataset rows: 75
- Dataset columns: 46
- Target distribution: {'Normal': 55, 'Mild Impairment': 20}
- Primary source described in README: GE-79 / PhysioNet CDED 1.0.1.
- Supporting reference dataset described in README: GE-75, retained for prospective external validation but excluded from the GE-79 modeling pipeline.

## Models 0-3

### Model 0 - Random Forest Feature Selection

- Purpose: select and lock the shared biomarker feature set for downstream model comparison.
- Output: `outputs/model0_FINAL_FEATURES.csv`.
- Supervised performance metrics: not applicable because Model 0 is a feature-selection step, not one of the final supervised classifiers.

### Model 1 - Logistic Regression

- Model type: linear probabilistic classifier.
- Role: explainable baseline model.
- Outputs include confusion matrix, ROC-AUC figure, PR-AUC figure, and metrics CSV.

### Model 2 - Decision Tree

- Model type: interpretable tree classifier.
- Role: nonlinear but readable comparison model.
- Outputs include tree visualization, confusion matrix, ROC-AUC figure, PR-AUC figure, and metrics CSV.

### Model 3 - Random Forest

- Model type: ensemble of decision trees.
- Role: nonlinear ensemble model with feature-importance and SHAP explainability outputs.
- Outputs include feature importance, confusion matrix, ROC-AUC, PR-AUC, SHAP global importance, SHAP summary, SHAP beeswarm, and SHAP waterfall figures.

## Features

Locked feature count: 14

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

Top Model 0 feature-ranking artifacts found:
- glucose_mg_dl
- fasting_glucose_mg_dl
- global_vasoreactivity
- daytime_sbp
- wmh_registered_masked
- wmh_registered
- ldl_calc_mg_dl
- svcam_ng_ml

Top Model 3 SHAP features found:
- daytime sbp
- fasting glucose mg dl
- glucose mg dl
- global vasoreactivity
- wmh registered

## Metrics

| Model | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall |
|---|---:|---:|---:|---:|---:|
| Model 1 - Logistic Regression | 0.547 | 0.525 | 0.532 | 0.508 | 0.5 |
| Model 2 - Decision Tree | 0.627 | 0.583 | 0.602 | 0.58 | 0.55 |
| Model 3 - Random Forest | 0.747 | 0.664 | 0.589 | 0.594 | 0.25 |

## ROC and PR

| Model | ROC-AUC | PR-AUC | Interpretation from Existing Report |
|---|---:|---:|---|
| Model 1 - Logistic Regression | 0.534 | 0.369 | Weak ranking performance, but stronger impaired recall than Random Forest. |
| Model 2 - Decision Tree | 0.639 | 0.370 | Moderate ROC ranking, but PR-AUC remains limited for the impaired minority class. |
| Model 3 - Random Forest | 0.648 | 0.441 | Best ROC-AUC and PR-AUC among the three, but impaired recall remains low. |

Available ROC/PR figure artifacts:

- Model 1 ROC-AUC: `outputs/model1_fig_roc_auc.png` (found)
- Model 1 PR-AUC: `outputs/model1_fig_pr_auc.png` (found)
- Model 2 ROC-AUC: `outputs/model2_fig_roc_auc.png` (found)
- Model 2 PR-AUC: `outputs/model2_fig_pr_auc.png` (found)
- Model 3 ROC-AUC: `outputs/model3_fig_roc_auc.png` (found)
- Model 3 PR-AUC: `outputs/model3_fig_pr_auc.png` (found)

## Responsible AI

- The project is documented as research-only.
- The NIST AI RMF assessment identifies intended use, dataset limitations, and risk controls.
- The PROBAST-AI report identifies high overall risk of bias due to small sample size, class imbalance, and lack of external validation.
- The TRIPOD-AI checklist reports 94.1% completeness from existing artifacts, with participant detail and data-cleaning audit gaps.
- VerifyWise was used as an independent governance screen across race, diabetes status, hypertension status, insulin use, and study group; it reported low bias for statistical parity, disparate impact, and demographic parity on the selected target.
- The VerifyWise result is interpreted narrowly because the analyzed target was the ground-truth `cognitive_status_label`, not model predictions; it does not prove the dataset or trained models are unbiased.
- Bias notes identify geographic, age, selection, survivorship, label, education, and racial representation risks.
- SHAP outputs are provided for Model 3 to support transparency, but they explain model behavior only and do not establish causality.

### VerifyWise Fairness Assessment

The cleaned GE-79 dataset was evaluated using the VerifyWise AI Bias Detector to assess statistical fairness across multiple protected attributes, including race, diabetes status, hypertension status, insulin use, and study group. VerifyWise reported low bias with statistical parity difference of 0.000, disparate impact ratio of 1.000, and demographic parity ratio of 1.000 for the tested comparisons. These results indicate that VerifyWise did not detect measurable parity differences under the selected metrics.

This should not be interpreted as proof that the dataset or models are unbiased. VerifyWise evaluated `cognitive_status_label`, which is a ground-truth clinical label rather than a model-generated decision. Model fairness would require evaluating model predictions by subgroup. Study-design risks remain, including geographic bias, selection bias, survivorship bias, small sample size, limited racial representation, label uncertainty, and lack of external validation. Accordingly, VerifyWise was used as one component of a broader Responsible AI assessment alongside documented dataset auditing, bias review, and governance practices informed by the NIST AI Risk Management Framework and OECD AI Principles.

## Limitations

- Small cohort size: 75 patients.
- Class imbalance: 55 No Impairment and 20 Impaired participants.
- No external validation output found.
- No clinical deployment evaluation found.
- Participant inclusion/exclusion criteria and detailed demographics are not fully represented in generated outputs.
- Confidence intervals, calibration metrics, and decision-curve analysis are not included in current outputs.
- Outcome labels may contain cognitive-testing label bias.
- Feature importance and SHAP values are interpretability tools, not causal evidence.

## Clinical Disclaimer

This project is not a medical device, diagnostic tool, clinical decision-support system, or patient-screening product. The models are research-only prototypes created for educational and exploratory analysis. They should not be used to diagnose cognitive impairment, guide treatment, prioritize care, or replace clinical judgment. External validation, calibration, clinical utility analysis, and expert clinical review would be required before any applied use.

## Model Selection Recommendation

Do not select a model based on accuracy alone. The existing outputs show that Model 3 has the highest accuracy but low impaired-class recall. If the project goal is screening sensitivity, impaired recall, macro F1, PR-AUC, and confusion matrices should be weighted more heavily than headline accuracy. The current best recommendation is to present all three supervised models as research comparisons, emphasize the class-imbalance tradeoff, and avoid deployment claims until an external validation study is completed.

## Recommended Report Portfolio

The generator creates `AI4ALL_ML-Diabetes_Ver_1_A/bias_reports/` and copies available generated reports into that folder. Missing report types are not fabricated.

Expected professional portfolio structure:

```text
AI4ALL_ML-Diabetes_Ver_1_A/
└── bias_reports/
    ├── NIST_AI_RMF_Report.md
    ├── TRIPOD_AI_Checklist.md
    ├── PROBAST_AI_Report.md
    ├── GE79_DATASET_CARD.md
    ├── GE79_MODEL_CARD.md
    ├── Responsible_AI_Report.md
    └── Project_Summary_Report.md
```

Currently generated by this script or already available:

- `NIST_AI_RMF_Report.md` copied from the existing NIST AI RMF assessment when found.
- `TRIPOD_AI_Checklist.md` copied when found.
- `PROBAST_AI_Report.md` copied when found.
- `GE79_MODEL_CARD.md` generated and copied.
- `SHAP_Report.md` copied when found.
- `model1_model2_model3_bias_mitigation.txt` copied when found.

## Notes

- This model card is generated from existing repository artifacts.
- Unknown or unavailable information is reported as a limitation rather than inferred.
- The report should be manually reviewed before formal submission or symposium use.
