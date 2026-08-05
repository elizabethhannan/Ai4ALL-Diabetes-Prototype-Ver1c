# GE-79 Cognitive-Status Classification: Diabetes and Cerebrovascular Biomarkers

AI4ALL Ignite 2026 · Group 6C

This project is a responsible machine-learning investigation of whether diabetes-related, cardiovascular, cerebrovascular, inflammatory, anthropometric, gait, and MRI-derived biomarkers from the GE-79 PhysioNet dataset can classify cognitive status in older adults. The project compares multiple supervised learning models while emphasizing healthcare AI practices that matter in real-world work: leakage prevention, feature selection, cross-validation, class-imbalance analysis, interpretability, dataset documentation, and bias/risk review.

Clinical status: this is a research and education project only. It is not validated for diagnosis, triage, treatment, screening, or clinical decision support.

## Table of Contents

- [Quick Links](#quick-links)
- [Research Question](#research-question)
- [Project Description and Evolution](#project-description-and-evolution)
- [Dataset](#dataset)
- [Algorithms, Inputs, and Outputs](#algorithms-inputs-and-outputs)
- [Evaluation Strategy](#evaluation-strategy)
- [Installation and Usage](#installation-and-usage)
- [Code Examples](#code-examples)
- [Visualizations and Supporting Materials](#visualizations-and-supporting-materials)
- [Responsible AI and Bias](#responsible-ai-and-bias)
- [Positive and Negative Impact](#positive-and-negative-impact)
- [Next Steps](#next-steps)
- [Repository Structure](#repository-structure)
- [Technologies Used](#technologies-used)
- [Contribution Guidelines](#contribution-guidelines)
- [Acknowledgments, Contacts, and References](#acknowledgments-contacts-and-references)

## Quick Links

- Model 0 Feature Selection: https://ai4all-diabetes-ml-model-0-features.streamlit.app/
- Model 1 Logistic Regression: https://ai4all-diabetes-app-ml-model-1-logistic-regression.streamlit.app/
- Model 2 Decision Tree: https://ai4all-diabetes-app-ml-model-2-decision-tree.streamlit.app/
- Model 3 Random Forest: https://ai4all-diabetes-app-ml-model-3-random-forest.streamlit.app/
- Bias & Responsible AI Reports: https://i4all-diabetes-ml-bias-report.streamlit.app/
- GitHub repository: https://github.com/elizabethhannan/AI4ALL-Diabetes-PRIVATE-ML

## Research Question

Can clinical, cardiovascular, glycemic, inflammatory, cerebrovascular, and MRI-derived biomarkers from the GE-79 cohort be used to classify mild cognitive impairment in older adults with Type 2 Diabetes, and how can the machine-learning workflow reduce or document bias during model development?

## Project Description and Evolution

The project began as a biomarker-based cognitive-status classification task and grew into a more complete responsible healthcare ML workflow. The final repository includes feature selection, three supervised classifiers, ROC-AUC and PR-AUC analysis, SHAP explainability for Random Forest, dataset auditing, VerifyWise fairness screening, NIST AI RMF assessment, OECD AI Principles mapping, TRIPOD-AI reporting review, PROBAST-AI risk-of-bias review, a Dataset Card, and a Model Card.

The central technical challenge is that the dataset is small and imbalanced: 55 participants are labeled Normal / No Impairment, and 20 participants are labeled Mild Impairment / Impaired. A model can appear successful on accuracy while still missing the minority impaired group. To avoid that mistake, the project evaluates macro F1, impaired recall, confusion matrices, ROC-AUC, PR-AUC, and Responsible AI documentation rather than relying on accuracy alone.

## Dataset

- Primary modeling dataset: GE-79 / PhysioNet CDED 1.0.1.
- Cleaned modeling file: `data/GE79_MASTER_DATASET_V1.csv`.
- Rows: 75.
- Columns: 46.
- Target: `cognitive_status_label`.
- Class distribution: 55 Normal / No Impairment and 20 Mild Impairment / Impaired.
- Supporting reference dataset: GE-75, retained for future external-validation planning but excluded from the current GE-79 modeling pipeline.

The dataset is documented in [GE79_DATASET_CARD.md](GE79_DATASET_CARD.md).

## Algorithms, Inputs, and Outputs

| Model | Algorithm Type | Inputs | Output | Pros | Cons |
|---|---|---|---|---|---|
| Model 0 | Random Forest feature selection | Candidate GE-79 biomarkers | Locked feature set | Reduces noisy predictors and standardizes downstream model inputs | Feature importance can be unstable in small samples |
| Model 1 | Logistic Regression, supervised linear classifier | Locked Model 0 biomarkers | Cognitive-status class probability and class label | Interpretable baseline; useful for comparison | Linear assumptions may miss nonlinear patterns |
| Model 2 | Decision Tree, supervised rule-based classifier | Locked Model 0 biomarkers | Cognitive-status class label | Human-readable decision rules | Can overfit small datasets |
| Model 3 | Random Forest, supervised ensemble classifier | Locked Model 0 biomarkers | Cognitive-status class probability, class label, feature importance | Stronger ensemble stability; supports feature-importance review | Less transparent than a single tree and had low impaired recall |

## Evaluation Strategy

All supervised classifiers use the same locked feature set and 5-fold stratified cross-validation. Preprocessing is handled inside scikit-learn pipelines to reduce leakage risk. Evaluation includes accuracy, precision, recall, macro F1-score, impaired recall, confusion matrix, ROC-AUC, and PR-AUC.

| Model | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Model 1 - Logistic Regression | 0.547 | 0.525 | 0.532 | 0.508 | 0.500 | 0.534 | 0.369 |
| Model 2 - Decision Tree | 0.627 | 0.583 | 0.602 | 0.580 | 0.550 | 0.639 | 0.370 |
| Model 3 - Random Forest | 0.747 | 0.664 | 0.589 | 0.594 | 0.250 | 0.648 | 0.441 |

Interpretation: Model 3 has the highest accuracy, ROC-AUC, and PR-AUC, but it misses many impaired participants. Model 2 has the strongest impaired recall. This supports the project's central lesson: in healthcare ML, accuracy alone is not enough.

## Installation and Usage

Prerequisites:

- Python 3.12.
- Git.
- A local copy of this repository.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the model scripts:

```bash
python3 src/model0_feature_selection.py
python3 src/model1_logistic_regression.py
python3 src/model2_decision_tree_complete.py
python3 src/model3_random_forest.py
```

Run a Streamlit dashboard locally:

```bash
streamlit run app/model0_streamlit.py
streamlit run app/model1_streamlit.py
streamlit run app/model2_streamlit.py
streamlit run app/model3_streamlit.py
```

Interpretation guidance:

- Use macro F1 and impaired recall when discussing model safety and class imbalance.
- Use ROC-AUC and PR-AUC to discuss ranking performance.
- Use confusion matrices to explain false positives and false negatives.
- Do not interpret feature importance or SHAP values as causal medical evidence.

## Code Examples

The core modeling workflow follows a scikit-learn pattern: prepare the locked feature matrix, keep preprocessing inside the pipeline, and evaluate with stratified cross-validation.

```python
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro", "recall_macro"])
```

This repository does not expose a public API. The main interfaces are Python scripts in `src/`, Streamlit dashboards in `app/`, and generated reports in the repository root and `outputs/`.

## Visualizations and Supporting Materials

The project includes more than three visuals, satisfying the AI4ALL final presentation and GitHub page visual requirements:

- Feature-selection and target-distribution visualizations.
- ECharts-based Streamlit visualizations for Models 0-3.
- Confusion matrices for Models 1-3.
- ROC-AUC and PR-AUC figures for Models 1-3.
- Decision Tree diagram.
- Random Forest feature importance.
- SHAP global importance, summary, beeswarm, and waterfall plots.
- VerifyWise, OECD, NIST, TRIPOD-AI, PROBAST-AI, Dataset Card, and Model Card summaries in the Bias Reports app.
- Full-width presentation export images at the bottom of each Streamlit app.

## Responsible AI and Bias

Creating AI/ML systems can amplify bias when teams treat data as neutral, optimize only for headline accuracy, ignore missingness, fail to inspect subgroup performance, or deploy models outside the population where they were developed. In this project, bias could be amplified by the small GE-79 sample, class imbalance, single-cohort design, limited racial representation, label uncertainty, and the risk that a model may miss impaired participants while still appearing accurate.

The project mitigates and documents these risks by excluding target-derived leakage features, using the same locked predictor set across models, applying stratified cross-validation, reporting minority-class recall and macro F1, adding ROC-AUC and PR-AUC, using SHAP for Random Forest transparency, documenting missing data, screening protected attributes with VerifyWise, mapping OECD AI Principles, applying NIST AI RMF governance, and creating dataset/model/risk-of-bias reports. These steps do not eliminate bias, but they make the limitations visible and reduce the chance of overstating the model's safety.

The `AI4ALL_ML-Diabetes_Ver_1_A/bias_reports/` folder and the Bias Reports Streamlit app include:

- NIST AI RMF assessment.
- OECD AI Principles mapping.
- VerifyWise fairness assessment.
- Fairlearn future-work section.
- TRIPOD-AI checklist.
- PROBAST-AI risk-of-bias report.
- GE-79 Dataset Card.
- GE-79 Model Card.
- SHAP explainability report.

## Positive and Negative Impact

Positive impact:

- The project investigates whether routinely collected biomarkers contain signal related to cognitive-status classification.
- Earlier research signals could help guide future studies on cognitive decline in Type 2 Diabetes.
- The project demonstrates responsible model evaluation beyond accuracy.

Negative impact and risk:

- A false negative could incorrectly classify an impaired participant as no impairment.
- A false positive could create unnecessary concern.
- The dataset is too small and limited for clinical deployment.
- Fairness cannot be fully established without subgroup evaluation of model predictions and external validation.

## Next Steps

| Timeline | Next Step | Measurable Output |
|---|---|---|
| July 2026 | Build the final Google Slides deck | Complete slide deck with research question, algorithms, visuals, model comparison, bias analysis, citations, and speaking roles |
| July 2026 | Build the final poster | Poster file summarizing the problem, data, methods, results, Responsible AI review, impact, and next steps |
| July 2026 | Build and polish the final GitHub Pages site | Public-facing GitHub page with visuals, citations, limitations, Streamlit links, and contact information |
| July 2026 | Save out-of-fold model predictions by participant | CSVs for Models 1-3 with prediction, probability, true label, and subgroup fields |
| July 2026 | Run Fairlearn MetricFrame analysis | Subgroup accuracy, recall, selection rate, false positive rate, and false negative rate |
| August 2026 | Prepare external-validation plan using GE-75 or another compatible cohort | Data dictionary comparison and feasibility memo |
| September 2026 | Reassess model selection priorities | Written recommendation weighting impaired recall, PR-AUC, and safety risk |

## Repository Structure

```text
app/                                      Streamlit apps for Models 0-3 and shared ECharts components
AI4ALL_ML-Diabetes_Ver_1_A/bias_reports/ Bias, safety, dataset, model, and rubric-aligned reports
data/                                     Cleaned GE-79 modeling dataset
docs/                                     Feature explanations
outputs/                                  Metrics, figures, SHAP outputs, and dashboard exports
src/                                      Model scripts
```

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- SHAP
- Streamlit
- streamlit-echarts
- Git / GitHub
- GitHub Pages

## Contribution Guidelines

Contributions should protect the scientific and Responsible AI standards of the project.

1. Open an issue or discussion before making major changes to the dataset, target definition, model-selection logic, or Responsible AI claims.
2. Keep preprocessing inside scikit-learn pipelines to reduce leakage risk.
3. Report metrics beyond accuracy, especially macro F1, impaired recall, confusion matrices, ROC-AUC, and PR-AUC.
4. Do not add clinical deployment language unless the model has external validation and expert clinical review.
5. Use clear Python naming, readable functions, and concise comments where the logic is not self-explanatory.
6. Include citations for new datasets, algorithms, frameworks, or clinical claims.

## Acknowledgments, Contacts, and References

Developed by AI4ALL Ignite 2026 Group 6C. This student project reflects AI4ALL's emphasis on technical excellence, responsible AI, and positive impact.

Contributors:

- Elizabeth Hannan
- Agastyya Kola

Instructor:

- Joyce D. Williams, PhD | Instructor at AI4ALL

Contributor contacts:

- Elizabeth Hannan: https://www.linkedin.com/in/elizabethhannan
- Agastyya Kola: https://www.linkedin.com/in/agastyya-kala-806197306/

Citations and data sources:

1. Novak, V., & Quispe, R. (2022). Cerebromicrovascular Disease in Elderly with Diabetes (version 1.0.1). PhysioNet. https://physionet.org/content/cded/1.0.1/
2. Novak, V., Zhao, P., Manor, B., Sejdic, E., Alsop, D., Abduljalil, A., Roberson, P. K., Munshi, M., & Novak, P. (2011). Adhesion molecules, altered vasoreactivity, and brain atrophy in type 2 diabetes. Diabetes Care, 34(11), 2438-2441. https://doi.org/10.2337/dc11-0969
3. Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220. https://physionet.org/
4. PhysioNet. Cerebral Perfusion in Diabetes 1.0.1 (GE-75 supporting reference dataset). https://physionet.org/content/cerebral-perfusion-diabetes/1.0.1/
5. National Institute of Standards and Technology. AI Risk Management Framework. https://www.nist.gov/itl/ai-risk-management-framework
6. OECD.AI. OECD AI Principles Overview. https://oecd.ai/en/ai-principles
7. scikit-learn. Cross-validation: evaluating estimator performance. https://scikit-learn.org/stable/modules/cross_validation.html
8. Fairlearn. Improve fairness of AI systems. https://fairlearn.org/
9. SHAP documentation. https://shap.readthedocs.io/en/latest/
10. TRIPOD+AI reporting guidance. https://www.tripod-statement.org/tripod-ai
11. PROBAST risk-of-bias tool. https://www.probast.org/

*[Ver.1B 7_09_26 EH]*
