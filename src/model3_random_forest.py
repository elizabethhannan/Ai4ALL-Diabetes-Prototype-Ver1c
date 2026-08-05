"""
============================================================================
 GE-79 / CDED  -  Group 6C  |  AI4ALL Ignite 2026
 MODEL 3 of 3:  RANDOM FOREST  (Ensemble + Feature Importance)
 Label stamped on all figures:  GE-79 · Random Forest · Ensemble
============================================================================
 WHERE THIS FILE GOES (PyCharm):
     AI4ALL-Diabetes-PRIVATE-ML-Ver1.B/
     └── src/
         └── model3_random_forest.py      <-- THIS FILE

 IT READS:
     ../data/GE79_MASTER_DATASET_V1.csv    (input dataset)
     ../outputs/model0_FINAL_FEATURES.csv  (locked biomarker features from Model 0)

 IT WRITES (to ../outputs/):
     model3_fig_rf_importance.png        - top-12 feature importance bar chart
     model3_fig_rf_confusion.png         - confusion matrix
     model3_fig_roc_auc.png              - ROC curve with ROC-AUC
     model3_fig_pr_auc.png               - Precision-Recall curve with PR-AUC
     model3_shap_global_importance.csv   - SHAP mean absolute feature impact
     model3_shap_patient_explanation.csv - SHAP feature impacts for one patient
     model3_fig_shap_global_importance.png - SHAP global feature-importance chart
     model3_fig_shap_summary.png         - SHAP summary bar plot
     model3_fig_shap_beeswarm.png        - SHAP beeswarm plot
     model3_fig_shap_waterfall.png       - SHAP waterfall plot for one patient
     model3_results_model3_forest.csv    - metrics row for the comparison table

 ----------------------------------------------------------------------------
 RUN ORDER:  Model 0 must run first (it creates model0_FINAL_FEATURES.csv).
             Then Model 1 and Model 3 can run in any order.
 ----------------------------------------------------------------------------

 WHAT THIS MODEL IS
   Random Forest = an ENSEMBLE of many decision trees. Each tree votes, and
   the majority vote is the prediction. Averaging many trees usually reduces
   the over-fitting a single tree suffers from, and the forest can rank how
   important each feature was across all its trees.

 TARGET ENCODING
   0 = No Impairment   (MMSE >= 28, "Normal")
   1 = Impaired        (MMSE 25-27)

 HONESTY NOTE (read the printed results)
   On this small, imbalanced dataset (n=75, 73% "No Impairment") the Random
   Forest tends to chase accuracy by predicting the majority class, which can
   drive minority-class (Impaired) recall toward 0. The majority-class
   baseline accuracy is 0.733 -- any model at or below that is not actually
   beating a constant guess. We report macro-F1 and Impaired recall so this
   is visible and honestly discussed, per the AI4ALL bias rubric.

 EVALUATION METRICS
   - Accuracy
   - Precision
   - Recall
   - Macro F1-score
   - Confusion Matrix
   - ROC Curve
   - ROC-AUC
   - Precision-Recall Curve
   - PR-AUC
   - SHAP Global Feature Importance
   - SHAP Individual Patient Explanation

 LEAKAGE CONTROL
   - MMSE (defines the label) is NOT a feature  -> no target leakage.
   - One row per patient (Visit 2)              -> no repeated-measures leakage.
   - Imputation + scaling happen INSIDE the CV folds via a Pipeline.

 NOTE ON COMMENTS: Python uses '#' for comments. Lines marked with '##' below
 are extra plain-language notes explaining WHY each step exists.
============================================================================
"""

## ---- imports: standard library ----
import runpy  ## lets Streamlit Cloud delegate to the lightweight app
import warnings  ## silence non-critical warnings
import sys  ## detect interactive vs automated runs
from pathlib import Path  ## OS-independent file paths


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
        app_path = app_dir / "model3_streamlit.py"
        runpy.run_path(str(app_path), run_name="__main__")
        st.stop()


_run_lightweight_streamlit_app_if_needed()

## ---- imports: data + math ----
import numpy as np  ## numeric helpers (dtype checks)
import pandas as pd  ## tables: read CSV, write results

## ---- imports: plotting (headless-safe) ----
import matplotlib

matplotlib.use("Agg")  ## "Agg" = save figures to file, no pop-up window
import matplotlib.pyplot as plt
import shap  ## model explainability: global + patient-level feature impacts

## ---- imports: scikit-learn machine-learning pieces ----
from sklearn.compose import ColumnTransformer  ## different prep for num vs cat cols
from sklearn.dummy import DummyClassifier  ## the "always guess majority" baseline
from sklearn.ensemble import RandomForestClassifier  ## THE model for this file
from sklearn.impute import SimpleImputer  ## fill in missing values
from sklearn.metrics import (accuracy_score,  ## overall % correct
                             average_precision_score,  ## area under precision-recall curve
                             classification_report,  ## per-class precision/recall/F1
                             confusion_matrix,  ## TP/FP/FN/TN grid
                             f1_score,  ## balance of precision & recall
                             precision_recall_curve,  ## PR curve points from probabilities
                             precision_score,  ## of predicted-positive, how many right
                             recall_score,  ## of actual-positive, how many caught
                             roc_auc_score,  ## ROC-AUC from predicted probabilities
                             roc_curve)  ## ROC curve points from probabilities
from sklearn.model_selection import StratifiedKFold, cross_val_predict  ## CV tools
from sklearn.pipeline import Pipeline  ## chains prep + model as one unit
from sklearn.preprocessing import OneHotEncoder, StandardScaler  ## encode + scale

## ---- import: our own neumorphic figure helpers (live in src/) ----
from neumorphic_visualizations import (
    save_confusion_matrix,  ## draws the styled confusion matrix
    save_model_comparison,  ## draws the Model 1 vs Model 3 comparison
    save_precision_recall_curve,  ## draws the PR-AUC curve
    save_roc_curve,  ## draws the ROC-AUC curve
    save_rf_importance,  ## draws the feature-importance bar chart
)

warnings.filterwarnings("ignore")  ## hide harmless convergence/version warnings
RANDOM_STATE = 42  ## fixed seed -> identical results every run

# ---------------------------------------------------------------------------
# 0. CONFIG
# ---------------------------------------------------------------------------
## PROJECT_ROOT = the folder ABOVE src/ (the repo root). parents[1] climbs one
## level up from this file, so paths work no matter where PyCharm is opened.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "GE79_MASTER_DATASET_V1.csv"  ## input data
FEATURES_PATH = PROJECT_ROOT / "outputs" / "model0_FINAL_FEATURES.csv"  ## written by Model 0
MODEL2_PATH = PROJECT_ROOT / "src" / "model2_decision_tree_complete.py"  ## checked for "is it done yet?"
OUT_DIR = PROJECT_ROOT / "outputs"  ## where results are saved
OUT_DIR.mkdir(parents=True, exist_ok=True)  ## create outputs/ if missing

## ---- figure-label standard: DATASET · MODEL · description (encoding) ----
DATASET = "GE-79"  ## dataset name stamped on every figure
MODEL_NAME = "Random Forest"  ## model name stamped on every figure
MODEL_ROLE = "Ensemble"  ## this model's role in the project
CLASS_0_LABEL = "No Impairment"  ## human-readable name for code 0
CLASS_1_LABEL = "Impaired"  ## human-readable name for code 1
ENCODING = f"0 = {CLASS_0_LABEL}, 1 = {CLASS_1_LABEL}"

TARGET_COL = "cognitive_status_label"  ## the column we predict
TARGET_MAP = {"Normal": 0, "Mild Impairment": 1}  ## text label -> numeric code

## Columns that must NEVER be used as predictors (ids, target copies, junk/empty cols)
DROP_COLS = ["patient_id", "datapatient_id", "visit", "cognitive_status_code",
             "cognitive_status_label", "gait_walk1_distance_m", "dm_status"]
ADD_MISSING_FLAG_FOR = ["diabetes_duration"]  ## 45% missing -> impute + add a 0/1 flag

NAVY, TEAL = "#1f3a5f", "#2a9d8f"  ## brand colors for figures


def caption(description):
    """DATASET · MODEL · description (encoding) -- consistent on every figure."""
    ## two-line title: name on top, role + encoding underneath
    return f"{DATASET} · {MODEL_NAME} — {description}\n{MODEL_ROLE} model  ({ENCODING})"


# ---------------------------------------------------------------------------
# MODEL 2 STATUS CHECK
# ---------------------------------------------------------------------------
def model2_is_placeholder():
    ## Returns True if Model 2 hasn't been completed by the student yet.
    ## A missing file, or one still holding the stub markers, counts as "not done".
    if not MODEL2_PATH.exists():
        return True
    text = MODEL2_PATH.read_text(encoding="utf-8")
    return "model = None" in text or "PLACEHOLDER" in text


def confirm_continue_with_model2_placeholder():
    ## Model 3 does NOT need Model 2 to run -- it only needs Model 0's feature list.
    ## This just warns that the 3-model comparison isn't complete yet.
    if not model2_is_placeholder():
        print("\nModel 2 status: implemented. It can be included in the final comparison.")
        return True

    note_lines = [
        "",
        "[!] Model 2 status: placeholder/student guide.",
        "    Model 3 can still run because it only depends on Model 0's locked features.",
        "    Model 2 is needed later only for the final three-model comparison table",
        "    and the Decision Tree visuals: model2_results_model2_tree.csv,",
        "    model2_fig_dt_tree.png, and model2_fig_dt_confusion.png.",
    ]
    print("\n".join(note_lines))
    ## Save the note to outputs/ so teammates see it even if they miss the console.
    (OUT_DIR / "model3_model2_placeholder_note.txt").write_text(
        "\n".join(line.strip() for line in note_lines if line.strip()) + "\n",
        encoding="utf-8")

    ## Interactive run -> let the user stop; automated run -> continue silently.
    if sys.stdin.isatty():
        answer = input("Continue running Model 3 without Model 2? [Y/n]: ").strip().lower()
        if answer in {"n", "no"}:
            print("Stopped before running Model 3.")
            return False
    else:
        print("    Continuing automatically because this is a non-interactive run.")

    return True


# ---------------------------------------------------------------------------
# 1. LOAD DATA + THE LOCKED FEATURE LIST
# ---------------------------------------------------------------------------
def load_data_and_features():
    df = pd.read_csv(DATA_PATH)  ## read the 75-patient dataset

    ## Safety check: the column we plan to flag must actually exist.
    missing_cols = [col for col in ADD_MISSING_FLAG_FOR if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required column(s): {missing_cols}")

    ## Add a 0/1 "was this value missing?" flag so missingness becomes a usable signal.
    for col in ADD_MISSING_FLAG_FOR:
        df[f"{col}_missing"] = df[col].isnull().astype(int)  ## 1 if missing, else 0

    y = df[TARGET_COL].map(TARGET_MAP)  ## text labels -> 0/1 target vector

    ## Reuse the EXACT features Model 0 locked, so all three models are comparable.
    if FEATURES_PATH.exists():
        final_features = pd.read_csv(FEATURES_PATH)["final_features"].tolist()
        locked_feature_count = len(final_features)
        ## include the missing-flag column if its parent feature is in the list
        for col in ADD_MISSING_FLAG_FOR:
            flag = f"{col}_missing"
            if col in final_features and flag in df.columns and flag not in final_features:
                final_features.append(flag)
        print(f"Loaded {locked_feature_count} locked features from {FEATURES_PATH.name}")
    else:
        raise FileNotFoundError(
            "model0_FINAL_FEATURES.csv not found. Run Model 0 "
            "(model0_feature_selection.py) FIRST.")

    ## Safety check: every locked feature must actually exist in the dataset.
    missing_features = [feature for feature in final_features if feature not in df.columns]
    if missing_features:
        raise ValueError(
            "model0_FINAL_FEATURES.csv contains feature(s) not present in the dataset: "
            f"{missing_features}. Re-run Model 0 and check the locked feature list.")

    X = df[final_features]  ## X = just the predictor columns
    print(f"Dataset: {len(df)} patients | model input columns used: {X.shape[1]}")
    print("Target:", df[TARGET_COL].value_counts().to_dict())  ## show class balance
    return X, y


# ---------------------------------------------------------------------------
# 2. PREPROCESSING (fit inside CV folds only -> no leakage)
# ---------------------------------------------------------------------------
def build_preprocessor(X):
    ## Separate columns by type so each gets the correct treatment.
    numeric = X.select_dtypes(include=np.number).columns.tolist()  ## number columns
    categorical = X.select_dtypes(include="object").columns.tolist()  ## text columns
    ## Numeric: median-fill blanks, then standardize to mean 0 / std 1.
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    ## Categorical: mode-fill blanks, then one-hot encode.
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(drop="first", handle_unknown="ignore"))])
    ## Bundle both so they run on the right columns automatically.
    return ColumnTransformer([("num", num_pipe, numeric),
                              ("cat", cat_pipe, categorical)])


# ---------------------------------------------------------------------------
# 3. THE RANDOM FOREST
#    n_estimators=400 : number of trees in the forest
#    max_depth=6       : limit each tree's depth (small dataset -> avoid overfit)
#    min_samples_leaf=3: a leaf must hold >=3 patients (smooths predictions)
#    class_weight=balanced: up-weights the minority "Impaired" class
# ---------------------------------------------------------------------------
def get_random_forest():
    ## Returns the forest with the settings fixed for this project.
    return RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# 4. EVALUATE WITH 5-FOLD STRATIFIED CROSS-VALIDATION
# ---------------------------------------------------------------------------
def evaluate(X, y):
    pre = build_preprocessor(X)  ## prep step from Section 2
    ## Stratified 5-fold keeps the 73/27 class ratio in every fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    ## majority-class baseline = the bar to beat (always "No Impairment" = 0.733).
    base = Pipeline([("pre", pre), ("clf", DummyClassifier(strategy="most_frequent"))])
    base_pred = cross_val_predict(base, X, y, cv=cv)

    ## the real model: preprocessing + random forest as one leakage-safe pipeline.
    rf = Pipeline([("pre", pre), ("clf", get_random_forest())])
    rf_pred = cross_val_predict(rf, X, y, cv=cv)  ## out-of-fold predictions for every patient
    rf_prob = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]

    ## helper that turns predictions into one row of metrics.
    def row(name, yp):
        return {"model": name,
                "accuracy": round(accuracy_score(y, yp), 3),
                "precision_macro": round(precision_score(y, yp, average="macro", zero_division=0), 3),
                "recall_macro": round(recall_score(y, yp, average="macro", zero_division=0), 3),
                "f1_macro": round(f1_score(y, yp, average="macro", zero_division=0), 3),
                "recall_impaired": round(recall_score(y, yp, pos_label=1, zero_division=0), 3)}

    ## build + save the two-row results table (baseline + random forest).
    results = pd.DataFrame([row("Baseline (majority)", base_pred),
                            row("Random Forest", rf_pred)])
    results.to_csv(OUT_DIR / "model3_results_model3_forest.csv", index=False)

    print("\n=== MODEL 3: Random Forest — 5-fold CV results ===")
    print(results.to_string(index=False))
    print("\nMajority-class baseline accuracy = 0.733 (always predicts 'No Impairment').")
    print("\nPer-class report (Random Forest):")
    print(classification_report(y, rf_pred,
                                target_names=[CLASS_0_LABEL, CLASS_1_LABEL], zero_division=0))

    ## Pull the Random Forest metrics so the figure footer uses the REAL numbers
    ## (never hard-coded). These come straight from the run above.
    rf_metrics = {
        "accuracy": round(accuracy_score(y, rf_pred), 3),
        "f1_macro": round(f1_score(y, rf_pred, average="macro", zero_division=0), 3),
        "recall_impaired": round(recall_score(y, rf_pred, pos_label=1, zero_division=0), 3),
    }
    ## count how many truly-impaired patients we caught vs missed (for the footer).
    impaired_total = int((y == 1).sum())
    impaired_caught = int(((y == 1) & (rf_pred == 1)).sum())
    impaired_missed = impaired_total - impaired_caught

    ## return predictions, the results table, AND the metrics needed for the caption.
    return rf_pred, rf_prob, results, rf_metrics, impaired_caught, impaired_missed, impaired_total


# ---------------------------------------------------------------------------
# 5. FIGURE A — FEATURE IMPORTANCE (the Random Forest's signature output)
# ---------------------------------------------------------------------------
def plot_feature_importance(X, y):
    pre = build_preprocessor(X)
    rf = Pipeline([("pre", pre), ("clf", get_random_forest())])
    rf.fit(X, y)  ## fit on full data for the ranking

    ## pull feature names + their importance scores out of the fitted forest.
    names = rf.named_steps["pre"].get_feature_names_out()
    importances = rf.named_steps["clf"].feature_importances_
    fi = (pd.DataFrame({"feature": names, "importance": importances})
          .sort_values("importance", ascending=False).head(12))  ## keep the top 12
    ## strip the encoder prefixes so labels read cleanly on the chart.
    fi["feature"] = (fi["feature"].str.replace("num__", "", regex=False)
                     .str.replace("cat__", "", regex=False))
    fi.to_csv(OUT_DIR / "model3_feature_importance_model3.csv", index=False)

    ## hand the table to the neumorphic helper to draw + save the bar chart.
    save_rf_importance(fi, OUT_DIR / "model3_fig_rf_importance.png")
    print("Saved model3_fig_rf_importance.png")


# ---------------------------------------------------------------------------
# 6. FIGURE B — CONFUSION MATRIX
#    The footer text is BUILT FROM THE REAL METRICS computed in evaluate(),
#    so the caption can never drift from the actual results.
# ---------------------------------------------------------------------------
def plot_confusion(y, y_pred, metrics, impaired_caught, impaired_missed, impaired_total):
    cm = confusion_matrix(y, y_pred)  ## 2x2 grid: rows = actual, cols = predicted
    ## Build the footer dynamically from the numbers evaluate() just computed.
    footer = (f"Accuracy {metrics['accuracy']:.3f} · macro-F1 {metrics['f1_macro']:.3f} · "
              f"impaired recall {metrics['recall_impaired']:.2f} — "
              f"misses {impaired_missed} of {impaired_total} impaired.")
    ## Hand everything to the neumorphic helper to draw + save the styled figure.
    save_confusion_matrix(
        cm, OUT_DIR / "model3_fig_rf_confusion.png",
        "Model 3 Confusion Matrix",
        "GE-79 · Random Forest · Ensemble",
        footer)
    print("Saved model3_fig_rf_confusion.png")
    print("Footer (from real metrics):", footer)


def plot_roc_and_pr(y, y_prob):
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    save_roc_curve(
        fpr,
        tpr,
        roc_auc,
        OUT_DIR / "model3_fig_roc_auc.png",
        3,
        "Random Forest (Ensemble)",
    )
    save_precision_recall_curve(
        recall,
        precision,
        pr_auc,
        OUT_DIR / "model3_fig_pr_auc.png",
        3,
        "Random Forest (Ensemble)",
    )
    print("Saved model3_fig_roc_auc.png")
    print("Saved model3_fig_pr_auc.png")
    print("\n" + "=" * 50)
    print("ROC-AUC RESULTS")
    print("=" * 50)
    print("\nModel 3 (Random Forest)")
    print(f"\nAUC = {roc_auc:.3f}")
    print(f"PR-AUC = {pr_auc:.3f}")


# ---------------------------------------------------------------------------
# 6a. FIGURE C — SHAP EXPLANATIONS
#    SHAP explains the fitted Random Forest without changing its parameters.
#    The model is fit once on the full locked feature set for interpretation,
#    mirroring the existing feature-importance chart.
# ---------------------------------------------------------------------------
def _clean_feature_name(name):
    ## SHAP receives transformed feature names; this makes chart labels readable.
    return (name.replace("num__", "")
            .replace("cat__", "")
            .replace("_", " "))


def _positive_class_shap_values(explainer, transformed_x):
    ## Different SHAP versions return binary-class values in different shapes.
    ## Normalize them so downstream plots always explain class 1 = Impaired.
    shap_values = explainer.shap_values(transformed_x)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        values = shap_values[1]
        base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    elif getattr(shap_values, "ndim", 0) == 3:
        values = shap_values[:, :, 1]
        base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
    else:
        values = shap_values
        base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value

    return np.asarray(values), float(base_value)


def _save_shap_global_plot(global_importance):
    ## Save a plain bar chart for reviewers who do not inspect the SHAP beeswarm.
    top = global_importance.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("white")
    ax.barh(top["feature"], top["mean_abs_shap"], color="#5279ad", edgecolor="#315f91")
    ax.set_title("GE-79 • Model 3 • Random Forest • SHAP Global Feature Importance",
                 fontsize=16, fontweight="bold", color="#0d1a3d", pad=12)
    ax.set_xlabel("Mean absolute SHAP value (average feature impact on impaired prediction)",
                  fontsize=11, fontweight="bold", color="#0d1a3d")
    ax.tick_params(axis="both", colors="#0d1a3d", labelsize=10)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model3_fig_shap_global_importance.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_shap_explanations(X, y):
    ## Fit one final forest on the full locked feature set for interpretation.
    pre = build_preprocessor(X)
    rf = Pipeline([("pre", pre), ("clf", get_random_forest())])
    rf.fit(X, y)

    ## SHAP works on the transformed matrix after imputation/scaling/encoding.
    transformed_x = rf.named_steps["pre"].transform(X)
    if hasattr(transformed_x, "toarray"):
        transformed_x = transformed_x.toarray()

    raw_feature_names = rf.named_steps["pre"].get_feature_names_out()
    feature_names = [_clean_feature_name(name) for name in raw_feature_names]
    transformed_df = pd.DataFrame(transformed_x, columns=feature_names)

    ## Build a SHAP explainer for the positive class and export global impacts.
    explainer = shap.TreeExplainer(rf.named_steps["clf"])
    shap_values, base_value = _positive_class_shap_values(explainer, transformed_x)
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=np.repeat(base_value, transformed_df.shape[0]),
        data=transformed_df.to_numpy(),
        feature_names=feature_names,
    )

    global_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    global_importance["average_direction"] = np.where(
        global_importance["mean_shap"] >= 0,
        "pushes toward Impaired prediction",
        "pushes toward No Impairment prediction",
    )
    global_importance.to_csv(OUT_DIR / "model3_shap_global_importance.csv", index=False)
    _save_shap_global_plot(global_importance)

    ## Use the highest predicted impaired probability as the individual example.
    predicted_prob = rf.predict_proba(X)[:, 1]
    patient_idx = int(np.argmax(predicted_prob))
    patient_explanation = pd.DataFrame({
        "patient_row_index": patient_idx,
        "actual_label": int(y.iloc[patient_idx] if hasattr(y, "iloc") else y[patient_idx]),
        "predicted_probability_impaired": predicted_prob[patient_idx],
        "feature": feature_names,
        "transformed_feature_value": transformed_df.iloc[patient_idx].to_numpy(),
        "shap_value": shap_values[patient_idx],
        "abs_shap_value": np.abs(shap_values[patient_idx]),
    }).sort_values("abs_shap_value", ascending=False)
    patient_explanation["contribution_direction"] = np.where(
        patient_explanation["shap_value"] >= 0,
        "increases impaired prediction",
        "decreases impaired prediction",
    )
    patient_explanation.to_csv(OUT_DIR / "model3_shap_patient_explanation.csv", index=False)

    plt.figure(figsize=(10, 7), facecolor="white")
    shap.summary_plot(shap_values, transformed_df, plot_type="bar",
                      max_display=14, show=False)
    plt.title("GE-79 • Model 3 • SHAP Summary Plot",
              fontsize=16, fontweight="bold", color="#0d1a3d", pad=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model3_fig_shap_summary.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    plt.figure(figsize=(10, 7), facecolor="white")
    shap.plots.beeswarm(shap_explanation, max_display=14, show=False)
    plt.title("GE-79 • Model 3 • SHAP Beeswarm Plot",
              fontsize=16, fontweight="bold", color="#0d1a3d", pad=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model3_fig_shap_beeswarm.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    plt.figure(figsize=(10, 7), facecolor="white")
    shap.plots.waterfall(shap_explanation[patient_idx], max_display=14, show=False)
    plt.title(
        f"GE-79 • Model 3 • SHAP Waterfall • Patient Row {patient_idx}",
        fontsize=16,
        fontweight="bold",
        color="#0d1a3d",
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "model3_fig_shap_waterfall.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print("Saved model3_shap_global_importance.csv")
    print("Saved model3_shap_patient_explanation.csv")
    print("Saved model3_fig_shap_global_importance.png")
    print("Saved model3_fig_shap_summary.png")
    print("Saved model3_fig_shap_beeswarm.png")
    print("Saved model3_fig_shap_waterfall.png")


# ---------------------------------------------------------------------------
# 6b. FIGURE D — MODEL 1 vs MODEL 3 COMPARISON  (only if Model 1 has run)
# ---------------------------------------------------------------------------
def plot_model_comparison(model3_results):
    ## the comparison needs Model 1's results CSV; skip gracefully if it's absent.
    model1_path = OUT_DIR / "model1_results_model1_logreg.csv"
    if not model1_path.exists():
        print("Skipped model1_model3_fig_model_comparison.png; run Model 1 first.")
        return
    model1_results = pd.read_csv(model1_path)
    ## hand both result tables to the neumorphic helper to draw the comparison.
    save_model_comparison(
        model1_results, model3_results,
        OUT_DIR / "model1_model3_fig_model_comparison.png")
    print("Saved model1_model3_fig_model_comparison.png")


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 64)
    print(" MODEL 3 of 3 — RANDOM FOREST  (GE-79 · Random Forest · Ensemble)")
    print("=" * 64)
    ## warn (but don't block) if Model 2 is still a student placeholder.
    if not confirm_continue_with_model2_placeholder():
        return
    X, y = load_data_and_features()  ## 1) load data + locked features
    ## 2) evaluate -> predictions, results table, and the real metrics for the footer
    y_pred, y_prob, results, rf_metrics, caught, missed, total = evaluate(X, y)
    plot_feature_importance(X, y)  ## 3) feature-importance chart
    ## 4) confusion matrix, captioned with the REAL computed numbers
    plot_confusion(y, y_pred, rf_metrics, caught, missed, total)
    plot_roc_and_pr(y, y_prob)  ## 5) ROC-AUC and PR-AUC figures
    plot_shap_explanations(X, y)  ## 6) SHAP explainability figures
    plot_model_comparison(results)  ## 7) Model 1 vs Model 3 comparison
    print("\nDone. Outputs written to ../outputs/")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# [Dictated by E. Hannan]
# ---------------------------------------------------------------------------
