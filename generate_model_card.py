"""
Generate a GE-79 model card from existing project artifacts.

This script does not train or retrain any model. It reads existing outputs and
writes GE79_MODEL_CARD.md. It also creates the recommended reports portfolio
folder and copies available generated reports into it.
"""

from __future__ import annotations

## ---- imports: standard library only; this script reads artifacts, not models ----
import csv
import shutil
from pathlib import Path


## ---- project paths and source artifacts used to build the model card ----
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"
README_PATH = PROJECT_ROOT / "README.md"
MODEL_CARD_PATH = PROJECT_ROOT / "GE79_MODEL_CARD.md"
DATASET_CARD_PATH = PROJECT_ROOT / "GE79_DATASET_CARD.md"
RUBRIC_ALIGNMENT_PATH = PROJECT_ROOT / "AI4ALL_RUBRIC_ALIGNMENT.md"
PORTFOLIO_REPORTS_DIR = PROJECT_ROOT / "AI4ALL_ML-Diabetes_Ver_1_A" / "bias_reports"

NIST_PATHS = [
    OUTPUTS_DIR / "NIST_AI_RMF_Assessment.md",
    PROJECT_ROOT / "NIST_AI_RMF_Assessment.md",
]
TRIPOD_PATH = PROJECT_ROOT / "TRIPOD_AI_Checklist.md"
PROBAST_PATH = PROJECT_ROOT / "PROBAST_AI_Report.md"
BIAS_PATH = OUTPUTS_DIR / "model1_model2_model3_bias_mitigation.txt"
SHAP_REPORT_PATH = PROJECT_ROOT / "SHAP_Report.md"


def read_text(path: Path) -> str:
    ## Missing optional artifacts are represented as empty text instead of crashing.
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    ## Return CSV rows; missing CSVs become an empty list for reporting.
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_nist_path() -> Path | None:
    ## The NIST report may live at the repo root or in outputs/, depending on run order.
    for path in NIST_PATHS:
        if path.exists():
            return path
    return None


def dataset_summary() -> dict[str, object]:
    ## Pull lightweight dataset facts directly from the cleaned GE-79 CSV.
    if not DATA_PATH.exists():
        return {"exists": False}

    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    target_counts: dict[str, int] = {}
    for row in rows:
        label = row.get("cognitive_status_label", "")
        target_counts[label] = target_counts.get(label, 0) + 1

    return {
        "exists": True,
        "rows": len(rows),
        "columns": len(reader.fieldnames or []),
        "target_counts": target_counts,
    }


def feature_list() -> list[str]:
    ## Model 0 writes the locked feature list used by Models 1-3.
    rows = read_csv_rows(OUTPUTS_DIR / "model0_FINAL_FEATURES.csv")
    return [row["final_features"] for row in rows if row.get("final_features")]


def top_model0_features(limit: int = 8) -> list[str]:
    ## Use the top feature-selection rows as model-card evidence.
    rows = read_csv_rows(OUTPUTS_DIR / "model0_feature_importance_fullscope.csv")
    names = []
    for row in rows[:limit]:
        feature = row.get("feature") or row.get("final_features") or ""
        if feature:
            names.append(feature)
    return names


def top_shap_features(limit: int = 5) -> list[str]:
    ## Use Model 3 SHAP outputs when they have already been generated.
    rows = read_csv_rows(OUTPUTS_DIR / "model3_shap_global_importance.csv")
    return [row["feature"] for row in rows[:limit] if row.get("feature")]


def model_metrics() -> dict[str, dict[str, str]]:
    ## Collect the latest metrics row for each model result file.
    sources = {
        "Model 1 - Logistic Regression": OUTPUTS_DIR / "model1_results_model1_logreg.csv",
        "Model 2 - Decision Tree": OUTPUTS_DIR / "model2_results_tree.csv",
        "Model 3 - Random Forest": OUTPUTS_DIR / "model3_results_model3_forest.csv",
    }
    metrics: dict[str, dict[str, str]] = {}
    for model_name, path in sources.items():
        rows = read_csv_rows(path)
        metrics[model_name] = rows[-1] if rows else {}
    return metrics


def extract_auc_rows() -> list[dict[str, str]]:
    ## Parse the ROC-AUC/PR-AUC table from the NIST report when available.
    nist_path = existing_nist_path()
    text = read_text(nist_path) if nist_path else ""
    if not text:
        return []

    rows: list[dict[str, str]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "| Model | ROC-AUC | PR-AUC | Interpretation |":
            for table_line in lines[idx + 2:]:
                if not table_line.startswith("|"):
                    break
                parts = [part.strip() for part in table_line.strip("|").split("|")]
                if len(parts) == 4:
                    rows.append(
                        {
                            "model": parts[0],
                            "roc_auc": parts[1],
                            "pr_auc": parts[2],
                            "interpretation": parts[3],
                        }
                    )
            break
    return rows


def metrics_table(metrics: dict[str, dict[str, str]]) -> list[str]:
    ## Format model metrics as markdown table rows.
    lines = [
        "| Model | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, row in metrics.items():
        if not row:
            lines.append(f"| {model_name} | Not found | Not found | Not found | Not found | Not found |")
            continue
        lines.append(
            f"| {model_name} | {row.get('accuracy', 'Not found')} | "
            f"{row.get('precision_macro', 'Not found')} | {row.get('recall_macro', 'Not found')} | "
            f"{row.get('f1_macro', 'Not found')} | {row.get('recall_impaired', 'Not found')} |"
        )
    return lines


def auc_table(auc_rows: list[dict[str, str]]) -> list[str]:
    ## Format AUC values as a markdown table, with a fallback row if absent.
    lines = [
        "| Model | ROC-AUC | PR-AUC | Interpretation from Existing Report |",
        "|---|---:|---:|---|",
    ]
    if not auc_rows:
        lines.append("| Models 1-3 | Not found | Not found | Exact values not found in a machine-readable report. |")
        return lines

    for row in auc_rows:
        lines.append(
            f"| {row['model']} | {row['roc_auc']} | {row['pr_auc']} | {row['interpretation']} |"
        )
    return lines


def artifact_status(path: Path) -> str:
    return "found" if path.exists() else "missing"


def portfolio_report_copies() -> list[str]:
    ## Copy generated reports into the Streamlit bias-report folder for presentation.
    PORTFOLIO_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    report_sources = [
        (existing_nist_path(), "NIST_AI_RMF_Report.md"),
        (TRIPOD_PATH, "TRIPOD_AI_Checklist.md"),
        (PROBAST_PATH, "PROBAST_AI_Report.md"),
        (DATASET_CARD_PATH, "GE79_DATASET_CARD.md"),
        (MODEL_CARD_PATH, "GE79_MODEL_CARD.md"),
        (SHAP_REPORT_PATH, "SHAP_Report.md"),
        (RUBRIC_ALIGNMENT_PATH, "AI4ALL_RUBRIC_ALIGNMENT.md"),
        (BIAS_PATH, "model1_model2_model3_bias_mitigation.txt"),
    ]

    for source, destination_name in report_sources:
        if source and source.exists():
            destination = PORTFOLIO_REPORTS_DIR / destination_name
            shutil.copyfile(source, destination)
            copied.append(str(destination.relative_to(PROJECT_ROOT)))

    return copied


def generate_model_card() -> str:
    ## Build the markdown model card from existing artifacts only.
    readme = read_text(README_PATH)
    data = dataset_summary()
    features = feature_list()
    model0_top = top_model0_features()
    shap_top = top_shap_features()
    metrics = model_metrics()
    auc_rows = extract_auc_rows()

    lines: list[str] = [
        "# GE-79 Model Card",
        "",
        "Generated from existing GE-79 project artifacts. This model card does not retrain models and does not introduce new analyses.",
        "",
        "## Model Card Summary",
        "",
        "- Project: GE-79 Cognitive-Status Classification.",
        "- Task: binary classification of cognitive status from diabetes and cerebrovascular biomarkers.",
        "- Target encoding: `0 = No Impairment`; `1 = Impaired`.",
        "- Project status: research-only educational machine learning project.",
        "- Clinical status: not validated for diagnosis, treatment, triage, or deployment.",
        "",
        "## Source Artifacts Used",
        "",
        f"- README: `README.md` ({artifact_status(README_PATH)})",
        f"- Dataset: `data/GE79_MASTER_DATASET_V1.csv` ({artifact_status(DATA_PATH)})",
        f"- Model 0 features: `outputs/model0_FINAL_FEATURES.csv` ({artifact_status(OUTPUTS_DIR / 'model0_FINAL_FEATURES.csv')})",
        f"- Model 1 metrics: `outputs/model1_results_model1_logreg.csv` ({artifact_status(OUTPUTS_DIR / 'model1_results_model1_logreg.csv')})",
        f"- Model 2 metrics: `outputs/model2_results_tree.csv` ({artifact_status(OUTPUTS_DIR / 'model2_results_tree.csv')})",
        f"- Model 3 metrics: `outputs/model3_results_model3_forest.csv` ({artifact_status(OUTPUTS_DIR / 'model3_results_model3_forest.csv')})",
        f"- NIST AI RMF assessment: `{existing_nist_path().relative_to(PROJECT_ROOT) if existing_nist_path() else 'Not found'}`",
        f"- TRIPOD-AI checklist: `TRIPOD_AI_Checklist.md` ({artifact_status(TRIPOD_PATH)})",
        f"- PROBAST-AI report: `PROBAST_AI_Report.md` ({artifact_status(PROBAST_PATH)})",
        f"- Bias mitigation notes: `outputs/model1_model2_model3_bias_mitigation.txt` ({artifact_status(BIAS_PATH)})",
        "",
        "## Intended Use",
        "",
        "The intended use is research and education: to evaluate whether diabetes-related, cardiovascular, inflammatory, cerebrovascular, and anthropometric biomarkers contain predictive signal for cognitive-status classification in the GE-79 cohort.",
        "",
        "Appropriate uses:",
        "",
        "- AI4ALL symposium presentation.",
        "- Model comparison and responsible AI discussion.",
        "- Exploratory biomarker-screening research.",
        "- Portfolio documentation for a healthcare ML workflow.",
        "",
        "Not appropriate uses:",
        "",
        "- Clinical diagnosis.",
        "- Treatment decisions.",
        "- Patient triage.",
        "- Automated screening deployment.",
        "- Generalization outside GE-79 without external validation.",
        "",
        "## Dataset",
        "",
        f"- Dataset rows: {data.get('rows', 'Not found')}",
        f"- Dataset columns: {data.get('columns', 'Not found')}",
        f"- Target distribution: {data.get('target_counts', 'Not found')}",
        "- Primary source described in README: GE-79 / PhysioNet CDED 1.0.1.",
        "- Supporting reference dataset described in README: GE-75, retained for prospective external validation but excluded from the GE-79 modeling pipeline.",
        "",
        "## Models 0-3",
        "",
        "### Model 0 - Random Forest Feature Selection",
        "",
        "- Purpose: select and lock the shared biomarker feature set for downstream model comparison.",
        "- Output: `outputs/model0_FINAL_FEATURES.csv`.",
        "- Supervised performance metrics: not applicable because Model 0 is a feature-selection step, not one of the final supervised classifiers.",
        "",
        "### Model 1 - Logistic Regression",
        "",
        "- Model type: linear probabilistic classifier.",
        "- Role: explainable baseline model.",
        "- Outputs include confusion matrix, ROC-AUC figure, PR-AUC figure, and metrics CSV.",
        "",
        "### Model 2 - Decision Tree",
        "",
        "- Model type: interpretable tree classifier.",
        "- Role: nonlinear but readable comparison model.",
        "- Outputs include tree visualization, confusion matrix, ROC-AUC figure, PR-AUC figure, and metrics CSV.",
        "",
        "### Model 3 - Random Forest",
        "",
        "- Model type: ensemble of decision trees.",
        "- Role: nonlinear ensemble model with feature-importance and SHAP explainability outputs.",
        "- Outputs include feature importance, confusion matrix, ROC-AUC, PR-AUC, SHAP global importance, SHAP summary, SHAP beeswarm, and SHAP waterfall figures.",
        "",
        "## Features",
        "",
        f"Locked feature count: {len(features)}",
        "",
    ]
    lines.extend(f"- `{feature}`" for feature in features)

    lines.extend(["", "Top Model 0 feature-ranking artifacts found:"])
    if model0_top:
        lines.extend(f"- {feature}" for feature in model0_top)
    else:
        lines.append("- Not found.")

    lines.extend(["", "Top Model 3 SHAP features found:"])
    if shap_top:
        lines.extend(f"- {feature}" for feature in shap_top)
    else:
        lines.append("- Not found.")

    lines.extend(["", "## Metrics", ""])
    lines.extend(metrics_table(metrics))

    lines.extend(["", "## ROC and PR", ""])
    lines.extend(auc_table(auc_rows))
    lines.extend(
        [
            "",
            "Available ROC/PR figure artifacts:",
            "",
            f"- Model 1 ROC-AUC: `outputs/model1_fig_roc_auc.png` ({artifact_status(OUTPUTS_DIR / 'model1_fig_roc_auc.png')})",
            f"- Model 1 PR-AUC: `outputs/model1_fig_pr_auc.png` ({artifact_status(OUTPUTS_DIR / 'model1_fig_pr_auc.png')})",
            f"- Model 2 ROC-AUC: `outputs/model2_fig_roc_auc.png` ({artifact_status(OUTPUTS_DIR / 'model2_fig_roc_auc.png')})",
            f"- Model 2 PR-AUC: `outputs/model2_fig_pr_auc.png` ({artifact_status(OUTPUTS_DIR / 'model2_fig_pr_auc.png')})",
            f"- Model 3 ROC-AUC: `outputs/model3_fig_roc_auc.png` ({artifact_status(OUTPUTS_DIR / 'model3_fig_roc_auc.png')})",
            f"- Model 3 PR-AUC: `outputs/model3_fig_pr_auc.png` ({artifact_status(OUTPUTS_DIR / 'model3_fig_pr_auc.png')})",
            "",
            "## Responsible AI",
            "",
            "- The project is documented as research-only.",
            "- The NIST AI RMF assessment identifies intended use, dataset limitations, and risk controls.",
            "- The PROBAST-AI report identifies high overall risk of bias due to small sample size, class imbalance, and lack of external validation.",
            "- The TRIPOD-AI checklist reports 94.1% completeness from existing artifacts, with participant detail and data-cleaning audit gaps.",
            "- VerifyWise was used as an independent governance screen across race, diabetes status, hypertension status, insulin use, and study group; it reported low bias for statistical parity, disparate impact, and demographic parity on the selected target.",
            "- The VerifyWise result is interpreted narrowly because the analyzed target was the ground-truth `cognitive_status_label`, not model predictions; it does not prove the dataset or trained models are unbiased.",
            "- Bias notes identify geographic, age, selection, survivorship, label, education, and racial representation risks.",
            "- SHAP outputs are provided for Model 3 to support transparency, but they explain model behavior only and do not establish causality.",
            "",
            "### VerifyWise Fairness Assessment",
            "",
            "The cleaned GE-79 dataset was evaluated using the VerifyWise AI Bias Detector to assess statistical fairness across multiple protected attributes, including race, diabetes status, hypertension status, insulin use, and study group. VerifyWise reported low bias with statistical parity difference of 0.000, disparate impact ratio of 1.000, and demographic parity ratio of 1.000 for the tested comparisons. These results indicate that VerifyWise did not detect measurable parity differences under the selected metrics.",
            "",
            "This should not be interpreted as proof that the dataset or models are unbiased. VerifyWise evaluated `cognitive_status_label`, which is a ground-truth clinical label rather than a model-generated decision. Model fairness would require evaluating model predictions by subgroup. Study-design risks remain, including geographic bias, selection bias, survivorship bias, small sample size, limited racial representation, label uncertainty, and lack of external validation. Accordingly, VerifyWise was used as one component of a broader Responsible AI assessment alongside documented dataset auditing, bias review, and governance practices informed by the NIST AI Risk Management Framework and OECD AI Principles.",
            "",
            "## Limitations",
            "",
            "- Small cohort size: 75 patients.",
            "- Class imbalance: 55 No Impairment and 20 Impaired participants.",
            "- No external validation output found.",
            "- No clinical deployment evaluation found.",
            "- Participant inclusion/exclusion criteria and detailed demographics are not fully represented in generated outputs.",
            "- Confidence intervals, calibration metrics, and decision-curve analysis are not included in current outputs.",
            "- Outcome labels may contain cognitive-testing label bias.",
            "- Feature importance and SHAP values are interpretability tools, not causal evidence.",
            "",
            "## Clinical Disclaimer",
            "",
            "This project is not a medical device, diagnostic tool, clinical decision-support system, or patient-screening product. The models are research-only prototypes created for educational and exploratory analysis. They should not be used to diagnose cognitive impairment, guide treatment, prioritize care, or replace clinical judgment. External validation, calibration, clinical utility analysis, and expert clinical review would be required before any applied use.",
            "",
            "## Model Selection Recommendation",
            "",
            "Do not select a model based on accuracy alone. The existing outputs show that Model 3 has the highest accuracy but low impaired-class recall. If the project goal is screening sensitivity, impaired recall, macro F1, PR-AUC, and confusion matrices should be weighted more heavily than headline accuracy. The current best recommendation is to present all three supervised models as research comparisons, emphasize the class-imbalance tradeoff, and avoid deployment claims until an external validation study is completed.",
            "",
            "## Recommended Report Portfolio",
            "",
            "The generator creates `AI4ALL_ML-Diabetes_Ver_1_A/bias_reports/` and copies available generated reports into that folder. Missing report types are not fabricated.",
            "",
            "Expected professional portfolio structure:",
            "",
            "```text",
            "AI4ALL_ML-Diabetes_Ver_1_A/",
            "└── bias_reports/",
            "    ├── NIST_AI_RMF_Report.md",
            "    ├── TRIPOD_AI_Checklist.md",
            "    ├── PROBAST_AI_Report.md",
            "    ├── GE79_DATASET_CARD.md",
            "    ├── GE79_MODEL_CARD.md",
            "    ├── AI4ALL_RUBRIC_ALIGNMENT.md",
            "    ├── Responsible_AI_Report.md",
            "    └── Project_Summary_Report.md",
            "```",
            "",
            "Currently generated by this script or already available:",
            "",
            "- `NIST_AI_RMF_Report.md` copied from the existing NIST AI RMF assessment when found.",
            "- `TRIPOD_AI_Checklist.md` copied when found.",
            "- `PROBAST_AI_Report.md` copied when found.",
            "- `GE79_DATASET_CARD.md` copied when found.",
            "- `GE79_MODEL_CARD.md` generated and copied.",
            "- `SHAP_Report.md` copied when found.",
            "- `AI4ALL_RUBRIC_ALIGNMENT.md` copied when found.",
            "- `model1_model2_model3_bias_mitigation.txt` copied when found.",
            "",
            "## Notes",
            "",
            "- This model card is generated from existing repository artifacts.",
            "- Unknown or unavailable information is reported as a limitation rather than inferred.",
            "- The report should be manually reviewed before formal submission or symposium use.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    model_card = generate_model_card()
    MODEL_CARD_PATH.write_text(model_card, encoding="utf-8")
    copied = portfolio_report_copies()
    print(f"Wrote {MODEL_CARD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote portfolio folder {PORTFOLIO_REPORTS_DIR.relative_to(PROJECT_ROOT)}")
    if copied:
        print("Copied reports:")
        for path in copied:
            print(f"- {path}")


if __name__ == "__main__":
    main()
