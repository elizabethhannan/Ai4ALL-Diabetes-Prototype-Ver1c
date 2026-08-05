"""
============================================================================
 GE-79 / CDED  -  Group 6C  |  AI4ALL Ignite 2026
 MODEL 1 of 3:  LOGISTIC REGRESSION  (Baseline)
 Label stamped on all figures:  GE-79 · Logistic Regression · Baseline
============================================================================
 WHERE THIS FILE GOES (PyCharm):
     Ai4ALL-Diabetes-Prototype-Ver1c/
     └── src/
         └── model1_logistic_regression.py      <-- THIS FILE

 IT READS:
     ../data/GE79_MASTER_DATASET_V1.csv    (input dataset)
     ../outputs/model0_FINAL_FEATURES.csv  (locked biomarker features from Model 0)

 IT WRITES (to ../outputs/):
     model1_fig_lr_confusion.png       - confusion matrix (neumorphic style)
     model1_fig_roc_auc.png            - ROC curve with ROC-AUC
     model1_fig_pr_auc.png             - Precision-Recall curve with PR-AUC
     model1_results_model1_logreg.csv  - metrics row for the comparison table

 ----------------------------------------------------------------------------
 RUN ORDER:  Run Model 0 FIRST (it creates model0_FINAL_FEATURES.csv). Then this file.
             Models 1, 2 and 3 can then run in any order.
 ----------------------------------------------------------------------------

 WHAT THIS MODEL IS
   Logistic Regression = a linear model that estimates the probability of the
   "Impaired" class from a weighted sum of the features. It is the simplest,
   most explainable model, so it serves as the BASELINE that the Decision Tree
   and Random Forest are compared against.

 TARGET ENCODING
   0 = No Impairment   (MMSE >= 28, "Normal")
   1 = Impaired        (MMSE 25-27)

 HONESTY NOTE
   The majority-class baseline scores 0.733 accuracy by always guessing
   "No Impairment." Logistic Regression with class_weight="balanced" gives up
   some raw accuracy in order to actually detect Impaired patients, so we
   report macro-F1 and Impaired recall, not just accuracy.

 EVALUATION METRICS
   - Accuracy
   - Precision
   - Recall
   - Macro F1-score
   - Confusion Matrix
   - ROC Curve
   - ROC-AUC

 LEAKAGE CONTROL
   - MMSE (defines the label) is NOT a feature  -> no target leakage.
   - One row per patient (Visit 2)              -> no repeated-measures leakage.
   - Imputation + scaling happen INSIDE the CV folds via a Pipeline.

 NOTE ON COMMENTS: Python uses '#' for comments. Lines marked with '##' below
 are extra plain-language notes explaining WHY each step exists.
============================================================================
"""

## ---- imports: standard library ----
import runpy                          ## lets Streamlit Cloud delegate to the lightweight app
import warnings                       ## lets us silence noisy non-critical warnings
import sys                            ## used to detect interactive vs automated runs
from pathlib import Path              ## builds file paths that work on any OS


def _run_lightweight_streamlit_app_if_needed():
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return

    if get_script_run_ctx(suppress_warning=True) is not None:
        app_dir = Path(__file__).resolve().parents[1] / "app"
        app_dir_str = str(app_dir)
        if app_dir_str in sys.path:
            sys.path.remove(app_dir_str)
        sys.path.insert(0, app_dir_str)
        loaded_helper = sys.modules.get("echarts_components")
        if loaded_helper is not None and not str(getattr(loaded_helper, "__file__", "")).startswith(app_dir_str):
            del sys.modules["echarts_components"]
        app_path = app_dir / "model1_streamlit.py"
        runpy.run_path(str(app_path), run_name="__main__")
        st.stop()


_run_lightweight_streamlit_app_if_needed()

## ---- imports: data + math ----
import numpy as np                    ## numeric helpers (dtype checks for prep)
import pandas as pd                   ## tables: read the CSV, write the results

## ---- imports: plotting (headless-safe) ----
import matplotlib
matplotlib.use("Agg")                 ## "Agg" = save figures to file, no pop-up window
import matplotlib.pyplot as plt

## ---- imports: scikit-learn machine-learning pieces ----
from sklearn.compose import ColumnTransformer        ## apply different prep to num vs cat cols
from sklearn.dummy import DummyClassifier            ## the "always guess majority" baseline
from sklearn.impute import SimpleImputer             ## fills in missing values
from sklearn.linear_model import LogisticRegression  ## THE model for this file
from sklearn.metrics import (accuracy_score,         ## overall % correct
                             average_precision_score, ## area under precision-recall curve
                             classification_report,  ## per-class precision/recall/F1
                             confusion_matrix,       ## TP/FP/FN/TN grid
                             f1_score,               ## balance of precision & recall
                             precision_recall_curve, ## PR curve points from probabilities
                             precision_score,        ## of predicted-positive, how many right
                             recall_score,           ## of actual-positive, how many caught
                             roc_auc_score,          ## ROC-AUC from predicted probabilities
                             roc_curve)              ## ROC curve points from probabilities
from sklearn.model_selection import StratifiedKFold, cross_val_predict  ## CV tools
from sklearn.pipeline import Pipeline                ## chains prep + model as one unit
from sklearn.preprocessing import OneHotEncoder, StandardScaler         ## encode + scale

## ---- import: our own neumorphic figure helper (lives in src/) ----
from neumorphic_visualizations import (
    save_confusion_matrix,           ## draws the styled confusion matrix
    save_precision_recall_curve,     ## draws the PR-AUC curve
    save_roc_curve,                  ## draws the ROC-AUC curve
)

warnings.filterwarnings("ignore")     ## hide harmless convergence/version warnings
RANDOM_STATE = 42                     ## fixed seed -> identical results every run (reproducible)

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
## PROJECT_ROOT = the folder ABOVE src/ (i.e. the repo root). parents[1] climbs
## one level up from this file, so paths work no matter where PyCharm is opened.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"            ## input data
FEATURES_PATH = PROJECT_ROOT / "outputs" / "model0_FINAL_FEATURES.csv"     ## written by Model 0
MODEL2_PATH = PROJECT_ROOT / "src" / "model2_decision_tree_complete.py"  ## checked for "is it done yet?"
OUT_DIR = PROJECT_ROOT / "outputs"                                         ## where results are saved
OUT_DIR.mkdir(parents=True, exist_ok=True)                                 ## create outputs/ if missing

## ---- figure-label standard: DATASET · MODEL · description (encoding) ----
DATASET = "GE-79"                     ## dataset name stamped on every figure
MODEL_NAME = "Logistic Regression"    ## model name stamped on every figure
MODEL_ROLE = "Baseline"               ## this model's role in the project
CLASS_0_LABEL = "No Impairment"       ## human-readable name for code 0
CLASS_1_LABEL = "Impaired"            ## human-readable name for code 1
ENCODING = f"0 = {CLASS_0_LABEL}, 1 = {CLASS_1_LABEL}"

TARGET_COL = "cognitive_status_label"                ## the column we predict
TARGET_MAP = {"Normal": 0, "Mild Impairment": 1}     ## text label -> numeric code

## Columns that must NEVER be used as predictors (ids, target copies, junk/empty cols)
DROP_COLS = ["patient_id", "datapatient_id", "visit", "cognitive_status_code",
             "cognitive_status_label", "gait_walk1_distance_m", "dm_status"]
ADD_MISSING_FLAG_FOR = ["diabetes_duration"]   ## 45% missing -> impute + add a 0/1 "was missing" flag

NAVY = "#1f3a5f"                       ## brand color used in figures


def caption(description):
    """DATASET · MODEL · description (encoding) -- consistent on every figure."""
    ## two-line title: name on top, role + encoding underneath
    return f"{DATASET} · {MODEL_NAME} — {description}\n{MODEL_ROLE} model  ({ENCODING})"


# ---------------------------------------------------------------------------
# MODEL 2 STATUS CHECK
# ---------------------------------------------------------------------------
def model2_is_placeholder():
    ## Returns True if Model 2 hasn't been completed by the student yet.
    ## We treat a missing file, or one still containing the stub markers, as "not done".
    if not MODEL2_PATH.exists():
        return True
    text = MODEL2_PATH.read_text(encoding="utf-8")
    return "model = None" in text or "PLACEHOLDER" in text


def confirm_continue_with_model2_placeholder():
    ## Model 1 does NOT need Model 2 to run -- it only needs Model 0's feature list.
    ## This function just warns the user that the 3-model comparison isn't ready yet.
    if not model2_is_placeholder():
        print("\nModel 2 status: implemented. It can be included in the final comparison.")
        return True

    ## Build a friendly note explaining what is (and isn't) blocked by Model 2.
    note_lines = [
        "",
        "[!] Model 2 status: placeholder/student guide.",
        "    Model 1 can still run because it only depends on Model 0's locked features.",
        "    Model 2 is needed later only for the final three-model comparison table",
        "    and the Decision Tree visuals: model2_results_model2_tree.csv,",
        "    model2_fig_dt_tree.png, and model2_fig_dt_confusion.png.",
    ]
    print("\n".join(note_lines))
    ## Save the same note to outputs/ so teammates see it even if they miss the console.
    (OUT_DIR / "model1_model2_placeholder_note.txt").write_text(
        "\n".join(line.strip() for line in note_lines if line.strip()) + "\n",
        encoding="utf-8")

    ## If a human is running it interactively, let them choose to stop; if it's an
    ## automated run (no terminal attached), just continue without asking.
    if sys.stdin.isatty():
        answer = input("Continue running Model 1 without Model 2? [Y/n]: ").strip().lower()
        if answer in {"n", "no"}:
            print("Stopped before running Model 1.")
            return False
    else:
        print("    Continuing automatically because this is a non-interactive run.")

    return True


# ---------------------------------------------------------------------------
# 1. LOAD DATA + THE LOCKED FEATURE LIST
# ---------------------------------------------------------------------------
def load_data_and_features():
    df = pd.read_csv(DATA_PATH)                 ## read the 75-patient dataset

    ## Safety check: make sure the column we want to flag actually exists.
    missing_cols = [col for col in ADD_MISSING_FLAG_FOR if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")

    ## Add a 0/1 "was this value missing?" flag so the model can use missingness as a signal.
    for col in ADD_MISSING_FLAG_FOR:
        df[f"{col}_missing"] = df[col].isnull().astype(int)   ## 1 if missing, else 0

    y = df[TARGET_COL].map(TARGET_MAP)          ## convert text labels -> 0/1 target vector

    ## Reuse the EXACT features Model 0 locked in, so all three models are comparable.
    if FEATURES_PATH.exists():
        final_features = pd.read_csv(FEATURES_PATH)["final_features"].tolist()
        locked_feature_count = len(final_features)
        ## If a flagged column made the list, make sure its _missing partner is included too.
        for col in ADD_MISSING_FLAG_FOR:
            flag = f"{col}_missing"
            if col in final_features and flag in df.columns and flag not in final_features:
                final_features.append(flag)
        print(f"Loaded {locked_feature_count} locked features from {FEATURES_PATH.name}")
    else:
        ## Stop early with a clear message if Model 0 hasn't been run yet.
        raise FileNotFoundError(
            "model0_FINAL_FEATURES.csv not found. Run Model 0 "
            "(model0_feature_selection.py) FIRST.")

    ## Safety check: every locked feature must actually exist in the dataset.
    missing_features = [feature for feature in final_features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "model0_FINAL_FEATURES.csv contains feature(s) not present in the dataset: "
            f"{missing_features}. Re-run Model 0 and check the locked feature list.")

    X = df[final_features]                       ## X = just the predictor columns
    print(f"Dataset: {len(df)} patients | model input columns used: {X.shape[1]}")
    print("Target:", df[TARGET_COL].value_counts().to_dict())   ## show class balance
    return X, y


# ---------------------------------------------------------------------------
# 2. PREPROCESSING (fit inside CV folds only -> no leakage)
# ---------------------------------------------------------------------------
def build_preprocessor(X):
    ## Separate columns by type so each gets the correct treatment.
    numeric = X.select_dtypes(include=np.number).columns.tolist()      ## number columns
    categorical = X.select_dtypes(include="object").columns.tolist()   ## text columns
    ## Numeric: fill blanks with the median, then standardize to mean 0 / std 1.
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    ## Categorical: fill blanks with the most common value, then one-hot encode.
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    ## Bundle both so they run on the right columns automatically.
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, categorical)])


# ---------------------------------------------------------------------------
# 3. THE LOGISTIC REGRESSION
#    class_weight="balanced": up-weights the minority "Impaired" class
#    max_iter=5000:           ensures the solver converges on scaled data
# ---------------------------------------------------------------------------
def get_logistic_regression():
    ## Returns the model with the settings fixed for this project.
    return LogisticRegression(max_iter=5000, class_weight="balanced")


# ---------------------------------------------------------------------------
# 4. EVALUATE WITH 5-FOLD STRATIFIED CROSS-VALIDATION
# ---------------------------------------------------------------------------
def evaluate(X, y):
    pre = build_preprocessor(X)                 ## prep step from Section 2
    ## Stratified 5-fold keeps the 73/27 class ratio in every fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    ## majority-class baseline = the bar to beat (always predicts "No Impairment" = 0.733).
    base = Pipeline([("pre", pre), ("clf", DummyClassifier(strategy="most_frequent"))])
    base_pred = cross_val_predict(base, X, y, cv=cv)

    ## the real model: preprocessing + logistic regression as one leakage-safe pipeline.
    lr = Pipeline([("pre", pre), ("clf", get_logistic_regression())])
    lr_pred = cross_val_predict(lr, X, y, cv=cv)   ## out-of-fold predictions for every patient
    lr_prob = cross_val_predict(lr, X, y, cv=cv, method="predict_proba")[:, 1]

    ## helper that turns predictions into one row of metrics.
    def row(name, yp):
        return {"model": name,
                "accuracy": round(accuracy_score(y, yp), 3),
                "precision_macro": round(precision_score(y, yp, average="macro", zero_division=0), 3),
                "recall_macro": round(recall_score(y, yp, average="macro", zero_division=0), 3),
                "f1_macro": round(f1_score(y, yp, average="macro", zero_division=0), 3),
                "recall_impaired": round(recall_score(y, yp, pos_label=1, zero_division=0), 3)}

    ## build + save the two-row results table (baseline + logistic regression).
    results = pd.DataFrame([row("Baseline (majority)", base_pred),
                            row("Logistic Regression", lr_pred)])
    results.to_csv(OUT_DIR / "model1_results_model1_logreg.csv", index=False)

    print("\n=== MODEL 1: Logistic Regression — 5-fold CV results ===")
    print(results.to_string(index=False))
    print("\nMajority-class baseline accuracy = 0.733 (always predicts 'No Impairment').")
    print("\nPer-class report (Logistic Regression):")
    print(classification_report(y, lr_pred,
          target_names=[CLASS_0_LABEL, CLASS_1_LABEL], zero_division=0))

    ## Pull out the Logistic Regression metrics so the figure footer can use the REAL
    ## numbers (never hard-coded). These come straight from the run above.
    lr_metrics = {
        "accuracy": round(accuracy_score(y, lr_pred), 3),
        "f1_macro": round(f1_score(y, lr_pred, average="macro", zero_division=0), 3),
        "recall_impaired": round(recall_score(y, lr_pred, pos_label=1, zero_division=0), 3),
    }
    ## count how many of the truly-impaired patients we actually caught (for the footer).
    impaired_total = int((y == 1).sum())
    impaired_caught = int(((y == 1) & (lr_pred == 1)).sum())

    ## return predictions AND the metrics needed to caption the figure honestly.
    return lr_pred, lr_prob, lr_metrics, impaired_caught, impaired_total


# ---------------------------------------------------------------------------
# 5. FIGURE — CONFUSION MATRIX
#    The footer text is BUILT FROM THE REAL METRICS computed in evaluate(),
#    so the caption can never drift from the actual results.
# ---------------------------------------------------------------------------
def plot_confusion(y, y_pred, metrics, impaired_caught, impaired_total):
    cm = confusion_matrix(y, y_pred)            ## 2x2 grid: rows = actual, cols = predicted
    ## Build the footer dynamically from the numbers evaluate() just computed.
    footer = (f"Accuracy {metrics['accuracy']:.3f} · macro-F1 {metrics['f1_macro']:.3f} · "
              f"impaired recall {metrics['recall_impaired']:.2f} — "
              f"catches {impaired_caught} of {impaired_total} impaired.")
    ## Hand everything to the neumorphic helper to draw + save the styled figure.
    save_confusion_matrix(
        cm, OUT_DIR / "model1_fig_lr_confusion.png",
        "Model 1 Confusion Matrix",
        "GE-79 · Logistic Regression · Baseline",
        footer)
    print("Saved model1_fig_lr_confusion.png")
    print("Footer (from real metrics):", footer)


def plot_roc_and_pr(y, y_prob):
    ## Convert predicted probabilities into ROC and Precision-Recall curve points.
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    ## Save standardized curve artifacts for the project outputs folder.
    save_roc_curve(
        fpr,
        tpr,
        roc_auc,
        OUT_DIR / "model1_fig_roc_auc.png",
        1,
        "Logistic Regression (Linear)",
    )
    save_precision_recall_curve(
        recall,
        precision,
        pr_auc,
        OUT_DIR / "model1_fig_pr_auc.png",
        1,
        "Logistic Regression (Linear)",
    )
    print("Saved model1_fig_roc_auc.png")
    print("Saved model1_fig_pr_auc.png")
    print("\n" + "=" * 50)
    print("ROC-AUC RESULTS")
    print("=" * 50)
    print("\nModel 1 (Logistic Regression)")
    print(f"\nAUC = {roc_auc:.3f}")
    print(f"PR-AUC = {pr_auc:.3f}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print(" MODEL 1 of 3 — LOGISTIC REGRESSION  (GE-79 · Logistic Regression · Baseline)")
    print("=" * 64)
    ## warn (but don't block) if Model 2 is still a student placeholder.
    if not confirm_continue_with_model2_placeholder():
        return
    X, y = load_data_and_features()             ## 1) load data + locked features
    ## 2) evaluate -> returns predictions plus the real metrics for the footer
    y_pred, y_prob, lr_metrics, impaired_caught, impaired_total = evaluate(X, y)
    ## 3) draw the confusion matrix using those real numbers
    plot_confusion(y, y_pred, lr_metrics, impaired_caught, impaired_total)
    plot_roc_and_pr(y, y_prob)
    print("\nDone. Outputs written to ../outputs/")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# [Dictated by E. Hannan]
# ---------------------------------------------------------------------------
