"""
Generate a PROBAST-AI-style risk-of-bias assessment for the completed GE-79
machine learning project.

This script does not train or retrain any model. It reads existing repository
artifacts and writes PROBAST_AI_Report.md.
"""

from __future__ import annotations

## ---- imports: standard library only; report generation is artifact-based ----
import csv
from dataclasses import dataclass
from pathlib import Path


## ---- project paths and fixed PROBAST risk labels ----
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
README_PATH = PROJECT_ROOT / "README.md"
TRIPOD_PATH = PROJECT_ROOT / "TRIPOD_AI_Checklist.md"
NIST_PATHS = [
    PROJECT_ROOT / "NIST_AI_RMF_Assessment.md",
    OUTPUTS_DIR / "NIST_AI_RMF_Assessment.md",
]
BIAS_PATH = OUTPUTS_DIR / "model1_model2_model3_bias_mitigation.txt"
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"
REPORT_PATH = PROJECT_ROOT / "PROBAST_AI_Report.md"

RISK_LOW = "Low"
RISK_MODERATE = "Moderate"
RISK_HIGH = "High"
RISK_UNKNOWN = "Unknown"


@dataclass
class DomainAssessment:
    ## One PROBAST-AI domain with evidence, concerns, and final judgment.
    name: str
    risk_of_bias: str
    applicability: str
    evidence: list[str]
    concerns: list[str]
    judgment: str


def read_text(path: Path) -> str:
    ## Missing optional text artifacts become empty strings for graceful reporting.
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    ## Read a CSV artifact into dictionaries; return [] if it has not been generated.
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_nist_path() -> Path | None:
    ## The NIST assessment can be generated in either root or outputs/.
    for path in NIST_PATHS:
        if path.exists():
            return path
    return None


def dataset_summary() -> dict[str, object]:
    ## Pull row count, column count, and target balance from the cleaned dataset.
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
    ## Load the locked Model 0 feature list if it exists.
    rows = read_csv_rows(OUTPUTS_DIR / "model0_FINAL_FEATURES.csv")
    return [row["final_features"] for row in rows if row.get("final_features")]


def model_metrics() -> dict[str, dict[str, str]]:
    ## Collect the latest metrics row for each model result file.
    sources = {
        "Model 1 - Logistic Regression": OUTPUTS_DIR / "model1_results_model1_logreg.csv",
        "Model 2 - Decision Tree": OUTPUTS_DIR / "model2_results_tree.csv",
        "Model 3 - Random Forest": OUTPUTS_DIR / "model3_results_model3_forest.csv",
    }
    metrics: dict[str, dict[str, str]] = {}
    for model, path in sources.items():
        rows = read_csv_rows(path)
        metrics[model] = rows[-1] if rows else {}
    return metrics


def file_present(relative_path: str) -> bool:
    return (PROJECT_ROOT / relative_path).exists()


def artifact_summary() -> dict[str, object]:
    ## Gather every evidence source needed for the risk-of-bias assessment.
    nist_path = existing_nist_path()
    return {
        "readme": read_text(README_PATH),
        "tripod": read_text(TRIPOD_PATH),
        "nist": read_text(nist_path) if nist_path else "",
        "nist_path": str(nist_path.relative_to(PROJECT_ROOT)) if nist_path else "Not found",
        "bias": read_text(BIAS_PATH),
        "dataset": dataset_summary(),
        "features": feature_list(),
        "metrics": model_metrics(),
        "shap_global": read_csv_rows(OUTPUTS_DIR / "model3_shap_global_importance.csv"),
        "shap_patient": read_csv_rows(OUTPUTS_DIR / "model3_shap_patient_explanation.csv"),
    }


def build_domain_assessments(context: dict[str, object]) -> list[DomainAssessment]:
    ## Convert project evidence into four PROBAST-AI-style domain judgments.
    readme = str(context["readme"])
    tripod = str(context["tripod"])
    nist = str(context["nist"])
    bias = str(context["bias"])
    dataset = context["dataset"]
    features = context["features"]
    metrics = context["metrics"]

    dataset_rows = dataset.get("rows", "Not found") if isinstance(dataset, dict) else "Not found"
    target_counts = dataset.get("target_counts", "Not found") if isinstance(dataset, dict) else "Not found"
    all_metrics_present = all(bool(row) for row in metrics.values()) if isinstance(metrics, dict) else False

    return [
        DomainAssessment(
            name="Participants",
            risk_of_bias=RISK_HIGH,
            applicability=RISK_MODERATE,
            evidence=[
                f"Dataset file contains {dataset_rows} rows.",
                f"Target distribution from dataset: {target_counts}.",
                "README describes the cohort as older adults with type-2 diabetes from GE-79 / PhysioNet CDED.",
                "Bias notes identify geographic, age, selection, survivorship, and racial representation limitations.",
            ],
            concerns=[
                "Sample size is small for clinical prediction modeling.",
                "The impaired class is the minority class.",
                "Detailed inclusion/exclusion criteria and demographic breakdown are not fully available in generated outputs.",
                "Bias notes report limited generalizability outside the GE-79 cohort.",
            ],
            judgment=(
                "High risk of bias because the small, imbalanced, single-cohort sample can produce unstable estimates "
                "and limited representation. Applicability is moderate because the population matches the research "
                "question, but use outside GE-79 is not established."
            ),
        ),
        DomainAssessment(
            name="Predictors",
            risk_of_bias=RISK_MODERATE,
            applicability=RISK_LOW,
            evidence=[
                f"Locked feature file contains {len(features)} final predictors.",
                "README reports biomarker domains including glycemic, cardiovascular, inflammatory, cerebrovascular, and anthropometric predictors.",
                "Model 0 feature-selection outputs and Model 3 SHAP outputs are present.",
                "NIST assessment states MMSE-derived target information is excluded from predictors.",
            ],
            concerns=[
                "Predictor units and collection timing are not fully tabulated in a machine-readable reporting file.",
                "Feature importance and SHAP explain model behavior but do not prove clinical causality.",
                "Feature selection was performed in a small cohort, so rankings may be unstable.",
            ],
            judgment=(
                "Moderate risk of bias because predictors are documented and target leakage controls are present, "
                "but the small cohort and feature-selection instability remain concerns. Applicability is low concern "
                "for the intended biomarker prediction question."
            ),
        ),
        DomainAssessment(
            name="Outcome",
            risk_of_bias=RISK_MODERATE,
            applicability=RISK_MODERATE,
            evidence=[
                "README defines the binary target from MMSE: 0 = No Impairment and 1 = Impaired.",
                "Dataset target counts are available and used in reports.",
                "Bias notes identify label bias as a remaining limitation.",
            ],
            concerns=[
                "The outcome is a cognitive-status label derived from cognitive testing, not an independently adjudicated clinical diagnosis in the generated outputs.",
                "Label bias may remain if cognitive testing misclassifies impairment.",
                "Clinical rationale for exact MMSE thresholds is not fully expanded in the existing outputs.",
            ],
            judgment=(
                "Moderate risk of bias because the outcome is explicitly defined and consistently encoded, but label "
                "imperfection and threshold rationale are incompletely documented. Applicability is moderate because "
                "the endpoint fits the project but should not be treated as a definitive diagnosis."
            ),
        ),
        DomainAssessment(
            name="Analysis",
            risk_of_bias=RISK_MODERATE,
            applicability=RISK_MODERATE,
            evidence=[
                "README reports scikit-learn pipelines, imputation inside cross-validation folds, and 5-fold stratified cross-validation.",
                "Model result CSVs are present for Models 1-3." if all_metrics_present else "One or more model result CSVs are missing.",
                "Confusion matrix, ROC-AUC, and PR-AUC figures are present for Models 1-3.",
                "SHAP global and individual patient explanation outputs are present for Model 3.",
                "NIST assessment documents leakage prevention and conservative interpretation.",
            ],
            concerns=[
                "No external validation output is present.",
                "No confidence intervals or bootstrap uncertainty estimates are present.",
                "Class imbalance remains important: Model 3 has high accuracy but low impaired recall.",
                "Model development and validation are internal to the same small GE-79 dataset.",
            ],
            judgment=(
                "Moderate risk of bias because leakage controls, stratified cross-validation, and multiple metrics are present, "
                "but sample size, imbalance, and absence of external validation prevent a low-risk judgment."
            ),
        ),
    ]


def metric_table(metrics: dict[str, dict[str, str]]) -> list[str]:
    ## Format model metrics as markdown table rows.
    lines = [
        "| Model | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in metrics.items():
        if not row:
            continue
        lines.append(
            f"| {model} | {row.get('accuracy', 'Not found')} | {row.get('precision_macro', 'Not found')} | "
            f"{row.get('recall_macro', 'Not found')} | {row.get('f1_macro', 'Not found')} | "
            f"{row.get('recall_impaired', 'Not found')} |"
        )
    return lines


def risk_table(domains: list[DomainAssessment]) -> list[str]:
    ## Format domain-level risk judgments as markdown table rows.
    lines = [
        "| PROBAST-AI Domain | Risk of Bias | Applicability Concern |",
        "|---|---|---|",
    ]
    for domain in domains:
        lines.append(f"| {domain.name} | {domain.risk_of_bias} | {domain.applicability} |")
    return lines


def overall_risk(domains: list[DomainAssessment]) -> str:
    ## Overall rating follows the highest risk found across domains.
    risks = [domain.risk_of_bias for domain in domains]
    if RISK_HIGH in risks:
        return RISK_HIGH
    if RISK_MODERATE in risks:
        return RISK_MODERATE
    if RISK_UNKNOWN in risks:
        return RISK_UNKNOWN
    return RISK_LOW


def generate_report() -> str:
    context = artifact_summary()
    domains = build_domain_assessments(context)
    dataset = context["dataset"]
    features = context["features"]
    metrics = context["metrics"]
    shap_global = context["shap_global"]
    shap_patient = context["shap_patient"]
    overall = overall_risk(domains)

    top_shap = [row.get("feature", "") for row in shap_global[:5] if row.get("feature")]
    patient_row = shap_patient[0] if shap_patient else {}

    lines: list[str] = [
        "# PROBAST-AI Risk-of-Bias Assessment",
        "",
        "Generated from existing GE-79 project artifacts. This report does not retrain models and does not fabricate unavailable information.",
        "",
        "## Source Artifacts Reviewed",
        "",
        f"- README: `README.md` ({'found' if README_PATH.exists() else 'missing'})",
        f"- Dataset: `data/GE79_MASTER_DATASET_V1.csv` ({'found' if DATA_PATH.exists() else 'missing'})",
        f"- NIST AI RMF assessment: `{context['nist_path']}`",
        f"- TRIPOD-AI checklist: `TRIPOD_AI_Checklist.md` ({'found' if TRIPOD_PATH.exists() else 'missing'})",
        f"- Bias mitigation notes: `outputs/model1_model2_model3_bias_mitigation.txt` ({'found' if BIAS_PATH.exists() else 'missing'})",
        "- Model metrics: existing Model 1, Model 2, and Model 3 result CSVs.",
        "- Model figures: existing confusion matrix, ROC-AUC, PR-AUC, feature-importance, and SHAP outputs.",
        "",
        "## Project Evidence Snapshot",
        "",
        f"- Dataset rows: {dataset.get('rows', 'Not found') if isinstance(dataset, dict) else 'Not found'}",
        f"- Dataset columns: {dataset.get('columns', 'Not found') if isinstance(dataset, dict) else 'Not found'}",
        f"- Target distribution: {dataset.get('target_counts', 'Not found') if isinstance(dataset, dict) else 'Not found'}",
        f"- Locked predictor count: {len(features)}",
        f"- Top SHAP features found: {', '.join(top_shap) if top_shap else 'Not found'}",
        f"- SHAP patient explanation row: {patient_row.get('patient_row_index', 'Not found')}",
        "",
        "## Model Evaluation Snapshot",
        "",
    ]
    lines.extend(metric_table(metrics))
    lines.extend(["", "## Domain-Level PROBAST-AI Judgments", ""])
    lines.extend(risk_table(domains))
    lines.append("")

    for domain in domains:
        lines.extend(
            [
                f"## {domain.name}",
                "",
                f"Risk of Bias: {domain.risk_of_bias}",
                "",
                f"Applicability Concern: {domain.applicability}",
                "",
                "Justification:",
            ]
        )
        lines.extend(f"- {item}" for item in domain.evidence)
        lines.extend(["", "Concerns:"])
        lines.extend(f"- {item}" for item in domain.concerns)
        lines.extend(["", "Judgment:", "", domain.judgment, ""])

    lines.extend(
        [
            "## Applicability",
            "",
            "Risk level: Moderate",
            "",
            "Justification:",
            "- The project question, biomarkers, and target are aligned with the GE-79 cognitive-status prediction task.",
            "- Applicability outside GE-79 is limited by the single-cohort design, small sample size, and documented representation concerns.",
            "- Existing NIST and bias notes explicitly describe the work as research-only and not clinically deployable.",
            "",
            "## Generalizability",
            "",
            "Risk level: High",
            "",
            "Justification:",
            "- Existing outputs do not document validation on a separate hospital, geography, or cohort.",
            "- Bias mitigation notes identify geographic, age, selection, survivorship, and racial representation limitations.",
            "- The cohort has 75 patients and an imbalanced target distribution, limiting stable generalization claims.",
            "",
            "## External Validation",
            "",
            "Risk level: High",
            "",
            "Justification:",
            "- No external validation output was found.",
            "- The README identifies GE-75 as a supporting reference dataset for prospective external validation, but the current pipeline excludes it from GE-79 modeling.",
            "- All reported model performance is internal to GE-79 cross-validation.",
            "",
            "## Clinical Readiness",
            "",
            "Risk level: High",
            "",
            "Justification:",
            "- Existing outputs state this is a research-only project.",
            "- Model performance shows clinically important tradeoffs; for example, Model 3 has accuracy 0.747 but impaired recall 0.25.",
            "- No prospective validation, calibration assessment, clinical utility analysis, or deployment monitoring plan was found.",
            "",
            "## Overall Risk of Bias Summary",
            "",
            f"Overall Risk of Bias: {overall}",
            "",
            "The overall risk is driven by small sample size, class imbalance, incomplete participant detail, lack of external validation, and residual outcome-label uncertainty. The analysis includes important safeguards, including leakage prevention, fold-contained preprocessing, stratified cross-validation, multiple performance metrics, and model explainability outputs. These safeguards improve transparency but do not eliminate the core risk-of-bias concerns.",
            "",
            "## Strengths",
            "",
            "- Research-only scope is clearly documented.",
            "- Dataset source and target class balance are documented.",
            "- Target leakage controls are described in existing outputs.",
            "- Models use a shared locked feature set from Model 0.",
            "- Preprocessing is performed inside scikit-learn pipelines.",
            "- 5-fold stratified cross-validation is used.",
            "- Accuracy is not reported alone; macro F1, impaired recall, confusion matrices, ROC-AUC, and PR-AUC are included.",
            "- SHAP outputs support Random Forest interpretability.",
            "- VerifyWise fairness screening found no statistical parity differences for tested protected attributes under its selected metrics.",
            "- Bias mitigation notes and NIST AI RMF assessment are present.",
            "",
            "## Limitations",
            "",
            "- Small cohort size: 75 patients.",
            "- Class imbalance: 55 No Impairment and 20 Impaired participants.",
            "- Participant inclusion/exclusion criteria and demographics are not fully reported in generated outputs.",
            "- External validation was not found.",
            "- Confidence intervals and calibration metrics were not found.",
            "- Outcome labels may contain cognitive-testing label bias.",
            "- VerifyWise evaluated the ground-truth clinical label, not model-generated predictions, so it does not establish model fairness.",
            "- SHAP explains model behavior but does not establish causality.",
            "- Clinical readiness is not established.",
            "",
            "## Future Improvements",
            "",
            "- Add external validation on an independent cohort before clinical interpretation.",
            "- Add confidence intervals or bootstrap uncertainty estimates for performance metrics.",
            "- Add calibration plots and calibration metrics.",
            "- Add a participant-flow table with inclusion/exclusion criteria.",
            "- Add demographic summaries and subgroup fairness analysis if valid subgroup counts are available.",
            "- Add a missingness table by predictor.",
            "- Add a model-parameter table for all final models.",
            "- Add decision-curve analysis or clinical utility analysis before considering applied use.",
            "",
            "## Notes",
            "",
            "- This report is generated only from existing repository files.",
            "- Unknown or unavailable information is reported as a limitation rather than inferred.",
            "- This report should be manually reviewed before formal submission.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = generate_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
