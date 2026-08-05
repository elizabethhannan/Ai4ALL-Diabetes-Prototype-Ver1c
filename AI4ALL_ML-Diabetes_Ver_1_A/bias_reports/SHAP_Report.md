# GE-79 SHAP Explainability Report

## Scope

This report summarizes the SHAP explainability outputs generated for Model 3, the Random Forest classifier. SHAP was used after model training to explain how the finalized Model 0 feature set contributed to Model 3 predictions. This report does not retrain the model and does not change preprocessing, feature selection, cross-validation, model parameters, or evaluation methodology.

## Generated Files

- Global feature importance CSV: `outputs/model3_shap_global_importance.csv`
- Individual patient explanation CSV: `outputs/model3_shap_patient_explanation.csv`
- Global feature importance figure: `outputs/model3_fig_shap_global_importance.png`
- Summary plot: `outputs/model3_fig_shap_summary.png`
- Beeswarm plot: `outputs/model3_fig_shap_beeswarm.png`
- Waterfall plot: `outputs/model3_fig_shap_waterfall.png`

## Model Explained

- Model: Model 3
- Machine learning type: Random Forest classifier
- Dataset: `GE79_MASTER_DATASET_V1.csv`
- Predictor set: finalized Model 0 feature set
- Outcome: cognitive impairment classification

## Global SHAP Feature Importance

The global SHAP output ranks predictors by mean absolute SHAP value. Higher mean absolute SHAP values indicate stronger average contribution to Model 3 predictions across the evaluated samples.

Top ranked features from `outputs/model3_shap_global_importance.csv`:

| Rank | Feature | Mean absolute SHAP |
|---:|---|---:|
| 1 | daytime sbp | 0.0467 |
| 2 | fasting glucose mg dl | 0.0441 |
| 3 | glucose mg dl | 0.0388 |
| 4 | global vasoreactivity | 0.0298 |
| 5 | wmh registered | 0.0275 |
| 6 | wmh registered masked | 0.0262 |
| 7 | nighttime sbp | 0.0261 |
| 8 | ldl calc mg dl | 0.0236 |
| 9 | perfusion whole brain baseline whole | 0.0227 |
| 10 | svcam ng ml | 0.0227 |

Technical:

- The strongest Model 3 explanations were associated with blood pressure, glucose-related measures, vascular reactivity, white matter hyperintensity measures, perfusion, lipid values, and endothelial/vascular markers.
- SHAP importance describes model behavior, not biological causality. A feature with a higher SHAP value had greater influence on the Random Forest predictions, but this does not prove it causes cognitive impairment.
- These outputs help audit whether Model 3 is relying on clinically plausible domains from the locked feature set rather than hidden or unintended inputs.
- Because the dataset is small and not externally validated, SHAP rankings should be interpreted as exploratory model-explanation evidence.

Takeaways:

- SHAP helps show which inputs most influenced the Random Forest model.
- The model appears to rely most on vascular, glucose, imaging, and cardiometabolic markers.
- This improves transparency because the project can explain why the model made predictions.
- These explanations are useful for review, but they are not clinical proof and should not be used for medical decisions.

## Individual Patient Explanation

The individual patient explanation file records feature-level SHAP contributions for one evaluated sample:

- Patient row index: 62
- Actual label: 1
- Predicted probability of impaired classification: 0.8374

Largest positive contributions toward the impaired prediction in this explanation included:

| Feature | SHAP value | Direction |
|---|---:|---|
| wmh registered | 0.0427 | increases impaired prediction |
| nighttime sbp | 0.0415 | increases impaired prediction |
| perfusion whole brain baseline whole | 0.0362 | increases impaired prediction |
| svcam ng ml | 0.0330 | increases impaired prediction |
| fasting glucose mg dl | 0.0327 | increases impaired prediction |

Technical:

- The waterfall explanation shows how individual feature contributions moved the Random Forest prediction for one sample.
- Positive SHAP values increased the predicted probability of cognitive impairment for that sample; negative SHAP values reduced it.
- The explanation supports model auditability by connecting one prediction to specific transformed feature values.
- Individual explanations are sample-specific and should not be generalized to all participants.

Takeaways:

- The waterfall plot explains one prediction step by step.
- For this sample, several vascular, imaging, and glucose-related features pushed the model toward an impaired prediction.
- This helps make the Random Forest less of a black box.
- It still does not prove a medical diagnosis or causal relationship.

## Responsible AI Interpretation

SHAP improves transparency for Model 3 by showing feature contributions at both global and individual levels. However, explainability does not remove known limitations of the GE-79 project, including small sample size, moderate class imbalance, single-site data, limited external validity, and the absence of prospective clinical validation.

Defensible use statement:

SHAP outputs may be used to support model interpretation, documentation, and Responsible AI review for this research project. They should not be described as proof that the model is clinically valid, unbiased, or ready for deployment.
