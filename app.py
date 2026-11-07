# ==============================================================================
# BITS Pilani (WILP) - M.Tech (AIML) Machine Learning Assignment 2
# Custom Credit-Risk Analytics & Multi-Model Evaluation Workbench
# Author: Sresth Anand | BITS ID: 2026AJ05039 | Random Seed: 5039
# Dataset: UCI Statlog German Credit Data (1,000 instances, 20 features)
# ==============================================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.special import expit

st.set_page_config(
    page_title="Sresth Anand (2026AJ05039) | Credit Risk ML Workbench",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom UI Styling (Section 5 UI-Level Originality & Customization)
st.markdown("""
<style>
    .sresth-banner {
        background: linear-gradient(135deg, #0f2c59 0%, #117864 100%);
        color: #ffffff;
        padding: 1.15rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 3px 10px rgba(15, 44, 89, 0.16);
    }
    .sresth-banner h2 { margin: 0; font-size: 1.5rem; font-weight: 700; color: #ffffff; }
    .sresth-banner p { margin: 0.25rem 0 0 0; font-size: 0.9rem; opacity: 0.92; }
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0f2c59;
        padding: 0.65rem 0.85rem;
        border-radius: 8px;
    }
</style>
<div class="sresth-banner">
    <h2>German Credit Risk Classification -- 5-Model Evaluation Workbench</h2>
    <p>Student: <b>Sresth Anand</b> | BITS ID: <b>2026AJ05039</b> | Seed: <b>5039</b> | Dataset: UCI Statlog German Credit (1,000 instances, 20 features)</p>
</div>
""", unsafe_allow_html=True)

SRESTH_MODEL_ARTIFACTS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl",
}

CREDIT_INPUT_ATTRIBUTES = [
    "checking_balance", "months_loan_duration", "credit_history", "purpose",
    "amount", "savings_balance", "employment_length", "installment_rate",
    "personal_status", "other_debtors", "residence_history", "property",
    "age", "installment_plan", "housing", "existing_credits",
    "dependents", "telephone", "foreign_worker", "job"
]

def encode_credit_dataframe(df_features, prep_artifact):
    encoded_blocks = []
    for col in prep_artifact["num_cols"]:
        arr_v = df_features[col].astype(float).values
        norm_v = (arr_v - prep_artifact["num_mean"][col]) / prep_artifact["num_std"][col]
        encoded_blocks.append(norm_v.reshape(-1, 1))
    for col in prep_artifact["cat_cols"]:
        str_v = df_features[col].astype(str).values
        for cat_lvl in prep_artifact["cat_categories"][col]:
            encoded_blocks.append((str_v == cat_lvl).astype(float).reshape(-1, 1))
    return np.hstack(encoded_blocks)

def _traverse_cart_node(row_vec, tree_node):
    while not tree_node["is_leaf"]:
        branch = "left" if row_vec[tree_node["feat"]] <= tree_node["thr"] else "right"
        tree_node = tree_node[branch]
    return tree_node["prob1"]

def evaluate_saved_credit_model(artifact_bundle, df_features):
    """Run inference using serialized model/*.pkl without retraining on uploaded test data."""
    if hasattr(artifact_bundle, "predict") and hasattr(artifact_bundle, "predict_proba"):
        preds = np.asarray(artifact_bundle.predict(df_features), dtype=int)
        probs = np.asarray(artifact_bundle.predict_proba(df_features)[:, 1], dtype=float)
        return preds, probs

    X_enc = encode_credit_dataframe(df_features, artifact_bundle["preprocessor"])
    arch = artifact_bundle["model_type"]

    if arch == "logistic_regression":
        prob_pos = expit(X_enc @ artifact_bundle["coef"] + artifact_bundle["intercept"])
    elif arch == "decision_tree":
        prob_pos = np.array([_traverse_cart_node(r, artifact_bundle["tree"]) for r in X_enc], dtype=float)
    elif arch == "knn":
        d_mat = cdist(X_enc, artifact_bundle["X_train"], metric="euclidean")
        nn_idx = np.argsort(d_mat, axis=1)[:, :int(artifact_bundle["n_neighbors"])]
        inv_w = 1.0 / (np.take_along_axis(d_mat, nn_idx, axis=1) + 1e-8)
        prob_pos = np.sum(inv_w * artifact_bundle["y_train"][nn_idx], axis=1) / np.sum(inv_w, axis=1)
    elif arch == "gaussian_nb":
        priors, mu, sigma2 = artifact_bundle["priors"], artifact_bundle["theta"], artifact_bundle["var"]
        log_lik = [
            np.log(priors[c]) - 0.5 * np.sum(np.log(2.0 * np.pi * sigma2[c]) + ((X_enc - mu[c]) ** 2) / sigma2[c], axis=1)
            for c in (0, 1)
        ]
        ll_mat = np.column_stack(log_lik)
        ll_mat -= np.max(ll_mat, axis=1, keepdims=True)
        prob_pos = (np.exp(ll_mat) / np.sum(np.exp(ll_mat), axis=1, keepdims=True))[:, 1]
    elif arch == "svm_rbf":
        rbf_k = np.exp(-artifact_bundle["gamma_val"] * cdist(X_enc, artifact_bundle["X_sv"], metric="sqeuclidean"))
        margin = rbf_k @ artifact_bundle["alpha"] + artifact_bundle["b"]
        prob_pos = expit(artifact_bundle["platt_a"] * margin + artifact_bundle["platt_b"])
        return (margin >= 0.0).astype(int), prob_pos
    else:
        raise ValueError(f"Unrecognized model architecture: {arch}")

    return (prob_pos >= 0.5).astype(int), prob_pos

def calculate_six_rubric_metrics(y_true, y_pred, y_prob):
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    ys = np.asarray(y_prob, dtype=float)

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    acc = (tp + tn) / max(len(yt), 1)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_den if mcc_den > 0 else 0.0

    s_pos, s_neg = ys[yt == 1], ys[yt == 0]
    pw_diff = s_pos[:, None] - s_neg[None, :]
    auc = float(np.sum(pw_diff > 0) + 0.5 * np.sum(pw_diff == 0)) / max(float(len(s_pos) * len(s_neg)), 1.0)

    return {
        "Accuracy": round(float(acc), 4), "AUC": round(float(auc), 4),
        "Precision": round(float(prec), 4), "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4), "MCC": round(float(mcc), 4),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

def generate_class_report_table(y_true, y_pred):
    yt, yp = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    report_rows, total_n = [], max(len(yt), 1)
    for cls_id, cls_label in [(0, "Class 0 (Good Credit)"), (1, "Class 1 (Default Risk)")]:
        tp = int(np.sum((yt == cls_id) & (yp == cls_id)))
        fp = int(np.sum((yt != cls_id) & (yp == cls_id)))
        fn = int(np.sum((yt == cls_id) & (yp != cls_id)))
        sup = int(np.sum(yt == cls_id))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0
        report_rows.append({"Class": cls_label, "Precision": round(p, 4), "Recall": round(r, 4), "F1-Score": round(f, 4), "Support": sup})
    report_rows.append({
        "Class": "Macro Average",
        "Precision": round(float(np.mean([r["Precision"] for r in report_rows[:2]])), 4),
        "Recall": round(float(np.mean([r["Recall"] for r in report_rows[:2]])), 4),
        "F1-Score": round(float(np.mean([r["F1-Score"] for r in report_rows[:2]])), 4),
        "Support": len(yt),
    })
    report_rows.append({
        "Class": "Weighted Average",
        "Precision": round(float(sum(r["Precision"] * r["Support"] for r in report_rows[:2]) / total_n), 4),
        "Recall": round(float(sum(r["Recall"] * r["Support"] for r in report_rows[:2]) / total_n), 4),
        "F1-Score": round(float(sum(r["F1-Score"] * r["Support"] for r in report_rows[:2]) / total_n), 4),
        "Support": len(yt),
    })
    return pd.DataFrame(report_rows)

@st.cache_resource
def load_serialized_models():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return {name: pickle.load(open(os.path.join(root_dir, rel_p), "rb")) for name, rel_p in SRESTH_MODEL_ARTIFACTS.items()}

loaded_models = load_serialized_models()

# Sidebar Controls: (a) Dataset Upload Option (CSV) & (b) Model Selection Dropdown
st.sidebar.header("Evaluation Controls (2026AJ05039)")
uploaded_csv = st.sidebar.file_uploader("a. Upload Held-Out Test CSV (20 features + target)", type=["csv"])

root_dir = os.path.dirname(os.path.abspath(__file__))
bundled_test_csv = os.path.join(root_dir, "test_data.csv")
if os.path.exists(bundled_test_csv):
    st.sidebar.download_button(
        "Download Bundled test_data.csv (N=200)",
        data=open(bundled_test_csv, "rb").read(),
        file_name="test_data.csv",
        mime="text/csv",
    )

active_model_name = st.sidebar.selectbox("b. Choose Classification Model", list(SRESTH_MODEL_ARTIFACTS.keys()), index=3)

if uploaded_csv is not None:
    eval_df = pd.read_csv(uploaded_csv)
    st.info(f"Evaluating uploaded test dataset `{uploaded_csv.name}` ({len(eval_df)} instances) using pre-trained `model/*.pkl` artifacts without retraining.")
else:
    eval_df = pd.read_csv(bundled_test_csv)
    st.info(f"Evaluating bundled held-out `test_data.csv` ({len(eval_df)} instances, Seed=5039). Upload any test CSV in the sidebar to evaluate dynamically.")

if "default" in eval_df.columns and "target" in eval_df.columns:
    eval_df = eval_df.drop(columns=["default"])

X_eval_df = eval_df[CREDIT_INPUT_ATTRIBUTES]
y_eval_true = eval_df["target"].astype(int).values

benchmark_rows, per_model_results = [], {}
for m_label, m_bundle in loaded_models.items():
    preds, probs = evaluate_saved_credit_model(m_bundle, X_eval_df)
    m_scores = calculate_six_rubric_metrics(y_eval_true, preds, probs)
    per_model_results[m_label] = {"y_pred": preds, "y_prob": probs, "metrics": m_scores}
    benchmark_rows.append({"ML Model Name": m_label, **{k: m_scores[k] for k in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}})

# (c) Display of Evaluation Metrics for Selected Model
chosen_res = per_model_results[active_model_name]
chosen_met = chosen_res["metrics"]
st.subheader(f"c. Evaluation Metrics -- {active_model_name}")
kpi_cols = st.columns(6)
for idx, (ui_lbl, m_key) in enumerate([
    ("Accuracy", "Accuracy"), ("AUC Score", "AUC"), ("Precision", "Precision"),
    ("Recall", "Recall"), ("F1 Score", "F1"), ("MCC Score", "MCC")
]):
    kpi_cols[idx].metric(ui_lbl, f"{chosen_met[m_key]:.4f}")

# (d) Confusion Matrix & Classification Report + 5-Model Comparison
tab_cm, tab_bench = st.tabs(["d. Confusion Matrix & Classification Report", "All 5 Models -- Benchmark Comparison on Test Data"])
with tab_cm:
    c_left, c_right = st.columns([1.0, 1.35])
    with c_left:
        cm_grid = np.array([[chosen_met["TN"], chosen_met["FP"]], [chosen_met["FN"], chosen_met["TP"]]])
        fig_cm, ax_cm = plt.subplots(figsize=(4.8, 3.6), dpi=130)
        sns.heatmap(
            cm_grid, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0 (Good)", "Pred 1 (Default)"],
            yticklabels=["True 0 (Good)", "True 1 (Default)"], ax=ax_cm
        )
        ax_cm.set_title(f"Confusion Matrix: {active_model_name}")
        st.pyplot(fig_cm)
    with c_right:
        st.dataframe(generate_class_report_table(y_eval_true, chosen_res["y_pred"]), use_container_width=True, hide_index=True)

with tab_bench:
    st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)