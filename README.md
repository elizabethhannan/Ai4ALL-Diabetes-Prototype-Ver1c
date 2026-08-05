# GE-79 Interactive Cognitive-Status Prototype — Ver1c

**AI4ALL Ignite · Summer Cohort 2026 · Group 6C**

This repository is the dedicated development space for an interactive, judge-facing prototype based on the GE-79 diabetes and cognitive-status machine-learning study.

The prototype is designed to demonstrate how two supervised classifiers respond to a partially constructed biomarker profile. It is a proof-of-concept research experience—not a diagnostic application, screening tool, or clinical decision-support system.

## Current Status

**Prototype Ver1c is under development.**

This repository currently contains a copied set of research scripts, documentation, aggregate model results, and visualization assets from the original GE-79 project. The separate React/D3 interactive application, prediction API, and deployable Replit configuration have not yet been added.

Accordingly, this README distinguishes between:

- **Verified research results already present in the repository**
- **Planned Prototype Ver1c functionality still to be implemented**

No current file should be interpreted as a completed clinical or consumer application.

## Research Question

Can clinical, cardiovascular, glycemic, inflammatory, cerebrovascular, and MRI-derived biomarkers from the GE-79 cohort be used to classify mild cognitive impairment in older adults with Type 2 Diabetes, and how can the machine-learning workflow reduce or document bias during model development?

## Prototype Purpose

Prototype Ver1c is intended for judges with limited demonstration time. Instead of requiring all 14 model inputs, the interface will allow a judge to choose three biomarkers and select a tested value for each.

> The remaining 11 features will be automatically populated using reference values derived from the GE-79 training data. These values are required because both models expect a complete 14-feature input profile.

The interface must always identify which three values came from the user and which eleven were supplied by the application. It must not present the result as an individualized medical assessment.

## Planned Judge Experience

1. Read the research question and project disclaimer.
2. Select one of the two models.
3. Choose three distinct biomarkers from the locked 14-feature set.
4. Select one tested, training-data-derived value for each biomarker.
5. Submit the completed profile.
6. Review the model classification, exploratory probability, explanation, and three dynamic visualizations.
7. Reset the experience for another demonstration.

Duplicate biomarker selections will be prevented. A **Reset** button will restore the model, biomarker selections, values, and results to their starting state.

## Model Selection

Select one of the two supervised machine-learning models below. Both models were trained using the same selected GE-79 clinical and MRI-derived biomarkers, allowing their performance and interpretability to be compared directly.

### Model 2 — Decision Tree

This model produces an interpretable, rule-based classification path. Its structure makes individual decisions easier to follow, although its predictions may be less stable when applied to a small dataset.

### Model 3 — Random Forest

This ensemble model combines predictions from multiple decision trees and achieved the strongest overall predictive performance in this study. SHAP is used to explain how individual features influenced its output.

These are model names—not “lower” and “higher” models. Either model may classify a completed profile as **Impaired** or **No Impairment**.

## Verified Model Results

All three supervised classifiers used the same locked feature set and 5-fold stratified cross-validation.

| Model | Accuracy | Macro F1 | Impaired Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Model 1 — Logistic Regression | 0.547 | 0.508 | 0.500 | 0.534 | 0.369 |
| Model 2 — Decision Tree | 0.627 | 0.580 | **0.550** | 0.639 | 0.370 |
| Model 3 — Random Forest | **0.747** | **0.594** | 0.250 | **0.648** | **0.441** |

Model 3 achieved the highest accuracy, macro F1, ROC-AUC, and PR-AUC. Model 2 identified a greater proportion of the impaired class. This distinction matters: Random Forest had the strongest overall performance, while Decision Tree had the stronger impaired recall. Accuracy alone would conceal that safety-relevant tradeoff.

## Planned Output

The selected model will return:

- **Model classification: Impaired** or **Model classification: No Impairment**
- **Exploratory probability assigned to the impaired class**
- A concise description of how to interpret the selected model
- Clear identification of the 3 user-selected and 11 auto-populated values
- Three dynamic, JavaScript-rendered visualizations
- A profile-level SHAP explanation where supported by the verified model pipeline

The probability is an exploratory model output. It is not a medically validated probability that a person has Mild Cognitive Impairment.

### Planned Dynamic Visualizations

1. **Classification probability** — the selected model’s output relative to its classification threshold.
2. **Feature contributions / SHAP** — which values moved the profile toward Impaired or No Impairment.
3. **Selected-feature sensitivity** — how the output changes across tested variations of the three selected biomarkers.

Model 2 may additionally display its rule-based decision path. Model 3 will emphasize SHAP contributions across the ensemble.

Existing confusion matrices, ROC and precision–recall curves, global feature importance, and aggregate SHAP images are study-level evidence. They may be reused as supporting assets, but they are not newly generated individual results.

## Planned Visual Design

The interface will carry the visual language of the AI4ALL presentation deck:

- Dark-evergreen background
- Mint-to-evergreen color palette
- Poppins typography
- Translucent, rounded content panels
- Responsive desktop and mobile layout
- Three animated result popouts after submission

A D3-powered particle brain will sit behind the input experience. Mint-colored particles will form a recognizable cerebrum, cerebellum, and brain stem; move away from mouse or touch interaction; and return slowly to their resting brain shape. The animation will include a reduced-motion fallback for accessibility.

The visual is an artistic interface element. Particle placement must not imply that a biomarker belongs to a specific anatomical brain region.

## Planned Technical Architecture

| Layer | Planned role |
|---|---|
| React | Judge-facing interface and state management |
| D3.js | Particle-brain animation and dynamic result graphics |
| Python prediction service | Loads the verified preprocessing and model artifacts |
| Model 2 | Decision Tree prediction and decision-path information |
| Model 3 | Random Forest prediction and SHAP-compatible output |
| Replit | Initial full-stack development and deployment environment |

D3 will visualize structured results returned by the Python model service; it will not replace or approximate the trained scikit-learn models.

## Scientific Guardrails

Before the interactive prediction flow is considered complete, the implementation must verify and freeze:

- Exact ordered list of 14 model features
- Feature units and valid input types
- Training-derived reference value for each auto-populated feature
- Tested dropdown values for judge-selected features
- Fitted preprocessing or imputation pipeline
- Saved Decision Tree and Random Forest artifacts
- Class encoding: `0 = No Impairment`, `1 = Impaired`
- Classification threshold
- SHAP-compatible background data and output shape
- Pretested profiles that produce accurate, reproducible visual results

Reference values and dropdown ranges must be derived from the GE-79 training data. They must not be invented simply to force a dramatic classification.

## Important Research Statement

> Returning to our original research question, our results provide evidence that supervised machine learning can classify Mild Cognitive Impairment in older adults with Type 2 Diabetes using clinical and MRI-derived biomarkers within the GE-79 dataset. Among the models evaluated, Random Forest achieved the strongest overall predictive performance, demonstrating that meaningful predictive relationships can be learned from clinical and MRI-derived biomarkers while remaining explainable through SHAP. However, these findings should be interpreted as a proof-of-concept and feasibility study rather than a clinically deployable diagnostic system.

## Clinical and Educational Disclaimer

This student prototype is presented for research demonstration, education, and entertainment purposes within the AI4ALL Summer Cohort 2026. It is not intended to diagnose, screen for, prevent, monitor, or treat any disease, ailment, cognitive condition, or other medical condition. Its outputs must not replace evaluation, diagnosis, or advice from a qualified healthcare professional.

The model was developed from a small, imbalanced research cohort of 75 participants: 55 labeled No Impairment and 20 labeled Impaired. It has not been externally validated for clinical use. False negatives may delay appropriate evaluation, while false positives may create unnecessary concern or testing.

## Repository Contents

The present repository primarily contains source research materials copied for prototype development, including:

```text
app/          Streamlit research dashboards
data/         GE-79 modeling data
docs/         Feature documentation
outputs/      Aggregate metrics, figures, and SHAP outputs
src/          Feature-selection and model-training scripts
```

These directories are research inputs and references. They are not yet the planned React/D3 prototype structure.

The prototype should eventually isolate its deployable interface, prediction service, versioned model artifacts, feature schema, reference values, and approved visual assets from the broader research materials.

## Research Dashboards

- [Model 0 — Feature Selection](https://ai4all-diabetes-ml-model-0-features.streamlit.app/)
- [Model 1 — Logistic Regression](https://ai4all-diabetes-app-ml-model-1-logistic-regression.streamlit.app/)
- [Model 2 — Decision Tree](https://ai4all-diabetes-app-ml-model-2-decision-tree.streamlit.app/)
- [Model 3 — Random Forest](https://ai4all-diabetes-app-ml-model-3-random-forest.streamlit.app/)
- [Bias and Responsible AI Reports](https://i4all-diabetes-ml-bias-report.streamlit.app/)

## Team and Acknowledgment

Developed by AI4ALL Ignite 2026 Group 6C:

- [Elizabeth Hannan](https://www.linkedin.com/in/elizabethhannan)
- [Agastyya Kola](https://www.linkedin.com/in/agastyya-kala-806197306/)

Special acknowledgment to **Professor Joyce D. Williams** for her instruction and guidance during the AI4ALL Summer Cohort.

## Dataset Citation

Novak, V., & Quispe, R. (2022). *Cerebromicrovascular disease in elderly with diabetes* (Version 1.0.1) [Data set]. PhysioNet. https://doi.org/10.13026/00bm-0x81

Additional study reference:

Novak, V., Zhao, P., Manor, B., Sejdic, E., Alsop, D., Abduljalil, A., Roberson, P. K., Munshi, M., & Novak, P. (2011). Adhesion molecules, altered vasoreactivity, and brain atrophy in type 2 diabetes. *Diabetes Care, 34*(11), 2438–2441. https://doi.org/10.2337/dc11-0969

---

**Prototype repository:** https://github.com/elizabethhannan/Ai4ALL-Diabetes-Prototype-Ver1c
