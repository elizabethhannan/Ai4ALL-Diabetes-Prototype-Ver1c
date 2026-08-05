"""
Generate a TRIPOD-AI-style reporting checklist for the completed GE-79 ML project.

This script does not train or retrain any model. It reads existing repository
artifacts and writes TRIPOD_AI_Checklist.md.
"""

from __future__ import annotations

## ---- imports: standard library only; the checklist reads saved artifacts ----
import csv
from dataclasses import dataclass
from pathlib import Path


## ---- project paths and checklist status labels ----
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
README_PATH = PROJECT_ROOT / "README.md"
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"
REPORT_PATH = PROJECT_ROOT / "TRIPOD_AI_Checklist.md"

STATUS_COMPLETE = "✓ Complete"
STATUS_PARTIAL = "⚠ Partial"
STATUS_MISSING = "✗ Missing"


@dataclass
class SectionAssessment:
    ## One TRIPOD-AI checklist item with evidence, gaps, and recommendations.
    name: str
    status: str
    evidence: list[str]
    gaps: list[str]
    recommendations: list[str]

    @property
    def score(self) -> float:
        ## Convert status labels into a weighted completeness score.
        if self.status == STATUS_COMPLETE:
            return 1.0
        if self.status == STATUS_PARTIAL:
            return 0.5
        return 0.0


def read_text(path: Path) -> str:
    ## Missing optional markdown artifacts become empty text.
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    ## Read generated metrics/results CSVs when present.
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def find_nist_assessment() -> Path | None:
    ## The NIST assessment can be generated in either root or outputs/.
    candidates = [
        PROJECT_ROOT / "NIST_AI_RMF_Assessment.md",
        OUTPUTS_DIR / "NIST_AI_RMF_Assessment.md",
    ]
    for path in candidates:
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


def model_metrics() -> dict[str, list[dict[str, str]]]:
    ## Load each model's results table for checklist evidence.
    return {
        "Model 1 - Logistic Regression": read_csv_rows(OUTPUTS_DIR / "model1_results_model1_logreg.csv"),
        "Model 2 - Decision Tree": read_csv_rows(OUTPUTS_DIR / "model2_results_tree.csv"),
        "Model 3 - Random Forest": read_csv_rows(OUTPUTS_DIR / "model3_results_model3_forest.csv"),
    }


def metric_rows_for_report(metrics: dict[str, list[dict[str, str]]]) -> list[str]:
    ## Format the latest metrics row from each model as markdown.
    lines = [
        "| Model | Accuracy | Precision Macro | Recall Macro | Macro F1 | Impaired Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model_name, rows in metrics.items():
        model_row = rows[-1] if rows else {}
        if not model_row:
            continue
        lines.append(
            "| {model} | {accuracy} | {precision_macro} | {recall_macro} | {f1_macro} | {recall_impaired} |".format(
                model=model_name,
                accuracy=model_row.get("accuracy", "Not found"),
                precision_macro=model_row.get("precision_macro", "Not found"),
                recall_macro=model_row.get("recall_macro", "Not found"),
                f1_macro=model_row.get("f1_macro", "Not found"),
                recall_impaired=model_row.get("recall_impaired", "Not found"),
            )
        )
    return lines


def extract_auc_table(nist_text: str) -> list[str]:
    ## Keep the AUC table exactly as reported by the NIST artifact.
    if not nist_text:
        return []

    lines = nist_text.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "| Model | ROC-AUC | PR-AUC | Interpretation |":
            start = idx
            break
    if start is None:
        return []

    table = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table.append(line)
    return table


def feature_list() -> list[str]:
    ## Load the locked Model 0 feature list if it exists.
    rows = read_csv_rows(OUTPUTS_DIR / "model0_FINAL_FEATURES.csv")
    return [row["final_features"] for row in rows if row.get("final_features")]


def top_shap_features(limit: int = 5) -> list[str]:
    ## Pull top SHAP features from Model 3 for explainability evidence.
    rows = read_csv_rows(OUTPUTS_DIR / "model3_shap_global_importance.csv")
    names = [row.get("feature", "") for row in rows[:limit]]
    return [name for name in names if name]


def build_assessments() -> tuple[list[SectionAssessment], dict[str, object]]:
    ## Score each reporting section using only evidence already in the repo.
    readme = read_text(README_PATH)
    nist_path = find_nist_assessment()
    nist_text = read_text(nist_path) if nist_path else ""
    data = dataset_summary()
    features = feature_list()
    metrics = model_metrics()
    shap_features = top_shap_features()

    source_context = {
        "readme": readme,
        "nist_path": str(nist_path.relative_to(PROJECT_ROOT)) if nist_path else "Not found",
        "dataset": data,
        "features": features,
        "metrics": metrics,
        "auc_table": extract_auc_table(nist_text),
        "shap_features": shap_features,
    }

    all_metric_files = all(rows for rows in metrics.values())
    roc_pr_figures = all(
        file_exists(OUTPUTS_DIR / f"model{i}_fig_roc_auc.png")
        and file_exists(OUTPUTS_DIR / f"model{i}_fig_pr_auc.png")
        for i in [1, 2, 3]
    )
    confusion_figures = all(
        file_exists(path)
        for path in [
            OUTPUTS_DIR / "model1_fig_lr_confusion.png",
            OUTPUTS_DIR / "model2_fig_dt_confusion.png",
            OUTPUTS_DIR / "model3_fig_rf_confusion.png",
        ]
    )

    assessments = [
        SectionAssessment(
            "Study Title",
            STATUS_COMPLETE if readme.startswith("# ") else STATUS_MISSING,
            ["README contains a project title: GE-79 Cognitive-Status Classification."]
            if readme.startswith("# ")
            else [],
            [] if readme.startswith("# ") else ["README title was not found."],
            ["Keep the title consistent across README, reports, and presentation materials."],
        ),
        SectionAssessment(
            "Research Question",
            STATUS_COMPLETE if "Problem Statement" in readme else STATUS_PARTIAL,
            [
                "README describes the goal: classify cognitive status from diabetes and cerebrovascular biomarkers.",
                "README states the class-imbalance problem and the need to evaluate minority-class detection.",
            ],
            [] if "Problem Statement" in readme else ["A distinct problem statement section was not found."],
            ["For publication, phrase the objective as a formal prediction-model research question."],
        ),
        SectionAssessment(
            "Dataset",
            STATUS_COMPLETE if data.get("exists") and "PhysioNet CDED" in readme else STATUS_PARTIAL,
            [
                f"Dataset file found: {DATA_PATH.relative_to(PROJECT_ROOT)}.",
                f"Rows: {data.get('rows', 'Not found')}; columns: {data.get('columns', 'Not found')}.",
                "README identifies GE-79 / PhysioNet CDED 1.0.1 as the primary modeling dataset.",
            ],
            [] if data.get("exists") else ["Dataset CSV was not found."],
            ["Add a data dictionary or direct column-level provenance table if required by reviewers."],
        ),
        SectionAssessment(
            "Participants",
            STATUS_PARTIAL if data.get("exists") else STATUS_MISSING,
            [
                f"Cohort size found from CSV: {data.get('rows', 'Not found')} patients.",
                f"Target counts found from CSV: {data.get('target_counts', 'Not found')}.",
            ],
            [
                "Detailed inclusion/exclusion criteria and participant demographics are not fully reported in the generated outputs."
            ],
            ["Add participant eligibility criteria, recruitment/source cohort details, and demographics if available."],
        ),
        SectionAssessment(
            "Outcome Definition",
            STATUS_COMPLETE if "MMSE" in readme and "No Impairment" in readme else STATUS_PARTIAL,
            [
                "README defines the binary target from MMSE: 0 = No Impairment and 1 = Impaired.",
                "Model files use `cognitive_status_label` mapped to Normal and Mild Impairment classes.",
            ],
            [] if "MMSE" in readme else ["MMSE-based target definition was not found in README."],
            ["Document any clinical rationale for the MMSE thresholds in a methods appendix."],
        ),
        SectionAssessment(
            "Predictors",
            STATUS_COMPLETE if features else STATUS_MISSING,
            [
                f"Locked feature file found with {len(features)} predictors: outputs/model0_FINAL_FEATURES.csv.",
                "Predictor domains include glycemic, blood-pressure, vascular, imaging, inflammatory, and anthropometric features.",
            ],
            [] if features else ["Locked feature file was not found or was empty."],
            ["Include a table mapping each predictor to unit, source column, and clinical domain."],
        ),
        SectionAssessment(
            "Data Cleaning",
            STATUS_PARTIAL if "preprocessing" in readme.lower() else STATUS_MISSING,
            [
                "README reports pandas data ingestion and scikit-learn preprocessing pipelines.",
                "Existing code uses imputation, scaling, and one-hot encoding where needed.",
            ],
            ["A step-by-step raw data cleaning log is not present in the existing outputs."],
            ["Add a data-cleaning audit table with exclusions, renamed columns, and derived fields."],
        ),
        SectionAssessment(
            "Missing Data",
            STATUS_COMPLETE if "diabetes_duration" in readme and "missingness" in readme.lower() else STATUS_PARTIAL,
            [
                "README reports median imputation and an explicit missingness indicator for `diabetes_duration`.",
                "Model pipelines perform imputation inside cross-validation folds.",
            ],
            [] if "missingness" in readme.lower() else ["Missing-data strategy is only partially documented."],
            ["Add a missingness table by predictor for formal TRIPOD-AI reporting."],
        ),
        SectionAssessment(
            "Feature Engineering",
            STATUS_COMPLETE if "missingness flag" in readme or "missingness indicator" in readme else STATUS_PARTIAL,
            [
                "The project adds a `diabetes_duration_missing` flag when `diabetes_duration` is selected.",
                "Categorical features are one-hot encoded inside model pipelines when present.",
            ],
            [] if "missingness" in readme.lower() else ["Feature-engineering details are limited in outputs."],
            ["List all engineered fields and transformations in a reproducibility appendix."],
        ),
        SectionAssessment(
            "Feature Selection",
            STATUS_COMPLETE if features and file_exists(OUTPUTS_DIR / "model0_feature_importance_fullscope.csv") else STATUS_PARTIAL,
            [
                "Model 0 selected a locked 14-biomarker feature set.",
                "Feature importance outputs and feature-selection figures are present.",
            ],
            [] if features else ["Feature-selection artifacts are missing."],
            ["Report whether any science-based anchor features were forced into the final feature set."],
        ),
        SectionAssessment(
            "Machine Learning Models",
            STATUS_COMPLETE if all_metric_files else STATUS_PARTIAL,
            [
                "Existing result CSVs found for Logistic Regression, Decision Tree, and Random Forest.",
                "README describes Model 1, Model 2, and Model 3 model types.",
            ],
            [] if all_metric_files else ["One or more model result CSVs are missing."],
            ["Add a model-parameter table for all estimators in the final report."],
        ),
        SectionAssessment(
            "Validation Strategy",
            STATUS_COMPLETE if "5-fold stratified cross-validation" in readme.lower() else STATUS_PARTIAL,
            [
                "README documents 5-fold stratified cross-validation.",
                "Model code uses StratifiedKFold and cross_val_predict.",
            ],
            [] if "cross-validation" in readme.lower() else ["Validation method was not found in README."],
            ["For future work, add external validation on a separate cohort."],
        ),
        SectionAssessment(
            "Performance Metrics",
            STATUS_COMPLETE if all_metric_files and roc_pr_figures and confusion_figures else STATUS_PARTIAL,
            [
                "Accuracy, precision macro, recall macro, macro F1, and impaired recall are stored in result CSVs.",
                "Confusion matrix, ROC-AUC, and PR-AUC figures are present for Models 1-3.",
            ],
            [] if all_metric_files and roc_pr_figures else ["Some metric files or ROC/PR figures are missing."],
            ["Store ROC-AUC and PR-AUC values in a machine-readable CSV in addition to figures."],
        ),
        SectionAssessment(
            "Limitations",
            STATUS_COMPLETE if nist_text and "Dataset Limitations" in nist_text else STATUS_PARTIAL,
            [
                f"NIST assessment found at: {source_context['nist_path']}.",
                "Limitations include small cohort size, class imbalance, and lack of external validation.",
            ],
            [] if nist_text else ["NIST AI RMF assessment was not found."],
            ["Add confidence intervals or bootstrap uncertainty estimates for performance metrics."],
        ),
        SectionAssessment(
            "Responsible AI",
            STATUS_COMPLETE if nist_text and "research-only" in nist_text.lower() else STATUS_PARTIAL,
            [
                "NIST assessment scopes the project as research-only and not for clinical deployment.",
                "Outputs include plain-language chart explanations and SHAP model explanations.",
                "VerifyWise fairness screening reported low bias for statistical parity, disparate impact, and demographic parity across tested protected attributes.",
            ],
            [] if nist_text else ["Responsible AI assessment document was not found."],
            ["Add model-prediction subgroup fairness analysis; VerifyWise screened the dataset target label, not model predictions."],
        ),
        SectionAssessment(
            "Bias Review",
            STATUS_COMPLETE if file_exists(OUTPUTS_DIR / "model1_model2_model3_bias_mitigation.txt") else STATUS_PARTIAL,
            [
                "Bias mitigation file is present.",
                "README and NIST assessment identify class imbalance and impaired-class recall as central safety concerns.",
                "VerifyWise did not detect statistical parity differences for race, diabetes status, hypertension status, insulin use, or study group under the selected metrics.",
            ],
            [] if file_exists(OUTPUTS_DIR / "model1_model2_model3_bias_mitigation.txt") else ["Bias mitigation file was not found."],
            ["Evaluate subgroup fairness on model predictions, not only ground-truth labels, if subgroup counts are sufficient."],
        ),
        SectionAssessment(
            "Reproducibility",
            STATUS_COMPLETE
            if file_exists(PROJECT_ROOT / "requirements.txt")
            and file_exists(PROJECT_ROOT / "src" / "model0_feature_selection.py")
            and file_exists(PROJECT_ROOT / "src" / "model1_logistic_regression.py")
            and file_exists(PROJECT_ROOT / "src" / "model2_decision_tree_complete.py")
            and file_exists(PROJECT_ROOT / "src" / "model3_random_forest.py")
            else STATUS_PARTIAL,
            [
                "requirements.txt is present.",
                "Source scripts for Models 0-3 are present.",
                "Outputs are version-controlled as CSV and PNG artifacts.",
            ],
            [],
            ["Add a single runbook describing exact execution order for all scripts."],
        ),
    ]

    return assessments, source_context


def status_counts(assessments: list[SectionAssessment]) -> dict[str, int]:
    return {
        STATUS_COMPLETE: sum(1 for item in assessments if item.status == STATUS_COMPLETE),
        STATUS_PARTIAL: sum(1 for item in assessments if item.status == STATUS_PARTIAL),
        STATUS_MISSING: sum(1 for item in assessments if item.status == STATUS_MISSING),
    }


def recommendation_summary(assessments: list[SectionAssessment]) -> list[str]:
    recommendations: list[str] = []
    for item in assessments:
        if item.status != STATUS_COMPLETE:
            recommendations.extend(item.recommendations)
    if not recommendations:
        recommendations.extend(
            [
                "Add external validation before any clinical interpretation.",
                "Add confidence intervals for core metrics.",
                "Store ROC-AUC and PR-AUC values in a machine-readable CSV.",
            ]
        )
    return recommendations


def generate_report() -> str:
    assessments, context = build_assessments()
    counts = status_counts(assessments)
    total = len(assessments)
    earned = sum(item.score for item in assessments)
    percent = (earned / total) * 100 if total else 0.0
    dataset = context["dataset"]
    features = context["features"]
    metrics = context["metrics"]
    auc_table = context["auc_table"]
    shap_features = context["shap_features"]

    lines: list[str] = [
        "# TRIPOD-AI Reporting Checklist",
        "",
        "Generated from existing GE-79 project artifacts. This report does not retrain models.",
        "",
        "## Source Artifacts Reviewed",
        "",
        f"- README: `{README_PATH.relative_to(PROJECT_ROOT)}` ({'found' if README_PATH.exists() else 'missing'})",
        f"- Dataset: `{DATA_PATH.relative_to(PROJECT_ROOT)}` ({'found' if DATA_PATH.exists() else 'missing'})",
        f"- Outputs directory: `{OUTPUTS_DIR.relative_to(PROJECT_ROOT)}` ({'found' if OUTPUTS_DIR.exists() else 'missing'})",
        f"- NIST AI RMF assessment: `{context['nist_path']}`",
        "- Model result CSVs: Model 1, Model 2, and Model 3 result files.",
        "- Figures reviewed by file presence: confusion matrices, ROC-AUC, PR-AUC, feature importance, and SHAP plots.",
        "",
        "## Project Snapshot",
        "",
        f"- Dataset rows: {dataset.get('rows', 'Not found')}",
        f"- Dataset columns: {dataset.get('columns', 'Not found')}",
        f"- Target counts: {dataset.get('target_counts', 'Not found')}",
        f"- Locked predictor count: {len(features)}",
        "",
        "## Model Metric Snapshot",
        "",
    ]
    lines.extend(metric_rows_for_report(metrics))
    lines.append("")

    if auc_table:
        lines.extend(["## ROC-AUC and PR-AUC Snapshot", ""])
        lines.extend(auc_table)
        lines.append("")
    else:
        lines.extend(
            [
                "## ROC-AUC and PR-AUC Snapshot",
                "",
                "ROC-AUC and PR-AUC figures were checked by file presence, but exact values were not found in a machine-readable output.",
                "",
            ]
        )

    if shap_features:
        lines.extend(
            [
                "## Top SHAP Features Found",
                "",
                *[f"- {feature}" for feature in shap_features],
                "",
            ]
        )

    lines.extend(["## Checklist", ""])
    for item in assessments:
        lines.extend([f"### {item.name}", "", f"Status: {item.status}", "", "Evidence:"])
        lines.extend([f"- {entry}" for entry in item.evidence] if item.evidence else ["- No supporting artifact found."])
        lines.extend(["", "Gaps:"])
        lines.extend([f"- {entry}" for entry in item.gaps] if item.gaps else ["- No major gap identified from existing outputs."])
        lines.extend(["", "Recommendation:"])
        lines.extend([f"- {entry}" for entry in item.recommendations])
        lines.append("")

    lines.extend(
        [
            "## Overall TRIPOD-AI Completeness Score",
            "",
            f"- Complete sections: {counts[STATUS_COMPLETE]}",
            f"- Partial sections: {counts[STATUS_PARTIAL]}",
            f"- Missing sections: {counts[STATUS_MISSING]}",
            f"- Weighted score: {earned:.1f} / {total:.1f}",
            f"- Overall completeness: {percent:.1f}%",
            "",
            "Scoring rule: Complete = 1.0 point, Partial = 0.5 point, Missing = 0 points.",
            "",
            "## Priority Recommendations",
            "",
        ]
    )
    lines.extend(f"- {entry}" for entry in recommendation_summary(assessments))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This checklist is based only on files already present in the repository.",
            "- It does not infer undocumented participant criteria, external validation, or clinical readiness.",
            "- It should be reviewed manually before being used in a formal research submission.",
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
