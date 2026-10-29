# M.Tech (AIML) — Machine Learning Assignment 2: End-to-End Credit Risk Classification & Streamlit Deployment

- **Student Name:** Sresth Anand
- **BITS ID:** `2026AJ05039`
- **Course:** Machine Learning (M.Tech AIML — Work Integrated Learning Programmes Division)
- **Random Seed:** `5039`

## a. Problem Statement

Commercial banks and financial institutions face substantial capital exposure when underwriting consumer loans. Misclassifying a high-risk borrower who subsequently defaults (*False Negative*) results in direct principal charge-offs, whereas misclassifying a creditworthy applicant (*False Positive*) incurs opportunity cost of foregone interest. The objective of this project is to design, evaluate, serialize, and deploy an end-to-end machine learning classification pipeline across five distinct supervised learning paradigms—**Logistic Regression**, **Decision Tree Classifier**, **K-Nearest Neighbor (KNN) Classifier**, **Gaussian Naive Bayes Classifier**, and **Support Vector Machine (SVM)**—to predict binary loan default risk (`target = 1` for Default Risk, `target = 0` for Good Credit) using 20 financial, demographic, and credit-history attributes, and to serve real-time test-set evaluation via an interactive **Streamlit Community Cloud** web application without retraining on uploaded test data.

## b. Dataset Description

- **Dataset Name:** Statlog (German Credit Data) Benchmark Dataset
- **Public Repository Source:** [UCI Machine Learning Repository — Statlog (German Credit Data), Dataset ID 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Total Instance Size:** **1,000 instances** (exceeds the minimum requirement of 500 instances; partitioned via an 80:20 stratified split using `SEED = 5039` into **800 training instances** and **200 held-out test instances** saved in `test_data.csv`).
- **Feature Size:** **20 input features** (exceeds the minimum requirement of 12 features), comprising **7 numerical attributes** (`months_loan_duration`, `amount`, `installment_rate`, `residence_history`, `age`, `existing_credits`, `dependents`) and **13 categorical attributes** (`checking_balance`, `credit_history`, `purpose`, `savings_balance`, `employment_length`, `personal_status`, `other_debtors`, `property`, `installment_plan`, `housing`, `telephone`, `foreign_worker`, `job`).
- **Target Variable (`target`):** Binary classification label where `0` = Creditworthy / Good Credit Risk (700 instances, 70.0% — 560 train / 140 test) and `1` = Default / Bad Credit Risk (300 instances, 30.0% — 240 train / 60 test).
- **Preprocessing Pipeline (`model/preprocessor.pkl`):** Zero missing values exist across all 1,000 records. The 7 continuous/integer attributes are standardized to zero mean and unit variance (`StandardScaler` fitted strictly on the 800 training rows), and the 13 categorical attributes are one-hot encoded with `drop_first=True` (`OneHotEncoder`), yielding a 48-dimensional numerical feature vector per instance.

## c. GitHub Repository Link & Live Streamlit App Link

- **GitHub Repository Link:** [https://github.com/2026aj05039-sresth-anand/ml-assignment2-credit-risk](https://github.com/2026aj05039-sresth-anand/ml-assignment2-credit-risk)
- **Live Streamlit App Link (Streamlit Community Cloud):** [https://2026aj05039-ml-assignment2.streamlit.app/](https://2026aj05039-ml-assignment2.streamlit.app/)

### Repository Structure (Step 3 Compliance)

```text
2026AJ05039-ml-assignment2/
|-- app.py                              # Interactive Streamlit web application (upload CSV, select model, view metrics & CM)
|-- train_models.py                     # End-to-end training, evaluation, and .pkl model serialization script
|-- 2026AJ05039_ML_Assignment2.ipynb    # Executed Jupyter Notebook from BITS Virtual Lab
|-- requirements.txt                    # Explicit Python package dependencies for Streamlit Community Cloud
|-- README.md                           # Project documentation, comparison table, and model performance observations
|-- test_data.csv                       # Held-out 200-row test dataset (20 input features + target column)
|-- german_credit.csv                   # Full 1,000-row UCI Statlog German Credit dataset
|-- model/                              # Saved pre-trained classifiers and preprocessing artifacts (.pkl)
    |-- preprocessor.pkl
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- svm.pkl
    |-- metrics_summary.json
```

## d. Models Used & Evaluation Metrics Comparison Table

All five classification models were trained on the identical 800-row training split (`SEED = 5039`) and evaluated on the held-out 200-row `test_data.csv` dataset across all six required metrics:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7800 | 0.7743 | 0.7353 | 0.4167 | 0.5319 | 0.4299 |
| **Decision Tree** | 0.7000 | 0.7005 | 0.5000 | 0.4167 | 0.4545 | 0.2520 |
| **KNN** | 0.7400 | 0.7425 | 0.6111 | 0.3667 | 0.4583 | 0.3181 |
| **Naive Bayes** | 0.7650 | **0.7985** | 0.5942 | **0.6833** | **0.6357** | **0.4659** |
| **Support Vector Machine** | **0.7800** | 0.7779 | **0.7500** | 0.4000 | 0.5217 | 0.4286 |

### Observations on Model Performance on the Chosen Dataset

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Ties for the highest overall **Accuracy (`0.7800`)** and achieves the second-highest **MCC (`0.4299`)** and **Precision (`0.7353`)**, with a strong **AUC (`0.7743`)** (`TN=131, FP=9, FN=35, TP=25`). L2 regularization (`C=0.5`) prevents overfitting across the 48 one-hot encoded credit features, yielding a well-calibrated linear log-odds boundary that is highly conservative on false alarms (only 9 false positives) but misses moderate-risk defaulters (`Recall = 0.4167`) due to the 70:30 class imbalance. |
| **Decision Tree** | Achieves the lowest performance among the five models (**Accuracy = `0.7000`**, **AUC = `0.7005`**, **Precision = `0.5000`**, **Recall = `0.4167`**, **F1 = `0.4545`**, **MCC = `0.2520`**; `TN=115, FP=25, FN=35, TP=25`). Even with depth regularization (`max_depth=10, min_samples_leaf=8`), a single axis-aligned CART tree suffers from high split variance when partitioning 48 mixed continuous and binary dummy features, producing coarser leaf probability estimates than smooth margin or probabilistic classifiers. |
| **KNN** | Delivers intermediate performance (**Accuracy = `0.7400`**, **AUC = `0.7425`**, **Precision = `0.6111`**, **Recall = `0.3667`**, **F1 = `0.4583`**, **MCC = `0.3181`**; `TN=126, FP=14, FN=38, TP=22`). Using distance-weighted Euclidean neighborhoods (`k=5`) improves local discrimination over uniform voting, but Euclidean distances in a 48-dimensional mixed one-hot/continuous space are diluted by irrelevant dummy dimensions (the curse of dimensionality) and majority-class crowding around borderline defaulters. |
| **Naive Bayes** | Achieves the **highest AUC (`0.7985`)**, **highest Recall (`0.6833`)**, **highest F1 Score (`0.6357`)**, and **highest MCC (`0.4659`)**, alongside strong **Accuracy (`0.7650`)** (`TN=112, FP=28, FN=19, TP=41`). By estimating class-conditional Gaussian likelihoods independently per feature (`var_smoothing=1e-2`), Gaussian Naive Bayes avoids over-parameterization and accumulates joint risk signals across checking balance, loan duration, and credit history—successfully catching **41 of 60 actual defaults** (versus only 22–25 for the other models). |
| **Support Vector Machine** | Ties for the **highest Accuracy (`0.7800`)**, achieves the **highest Precision (`0.7500`)** across all models, and records the second-highest **AUC (`0.7779`)** with **F1 = `0.5217`** and **MCC = `0.4286`** (`TN=132, FP=8, FN=36, TP=24`). The RBF kernel (`C=0.5, gamma=0.03`) captures non-linear interactions between loan amount, duration, and savings balance while maintaining the lowest false-positive rate (only 8 false alarms out of 140 good loans), though its standard `0.0` margin threshold favors specificity over sensitivity (`Recall = 0.4000`). |
| **Overall, Winner for your dataset?** | **Gaussian Naive Bayes** is the **Overall Winner** for credit-risk classification on this dataset, ranking **#1 in 4 out of 6 evaluation metrics** (**AUC = `0.7985`**, **MCC = `0.4659`**, **F1 = `0.6357`**, and **Recall = `0.6833`**) and within `1.5%` of the top Accuracy (`0.7650` vs. `0.7800`). Because the UCI German Credit dataset has a 70:30 class imbalance where missing a defaulting borrower (*False Negative*) is financially far costlier than a secondary review (*False Positive*), **MCC** and **AUC/F1** are the most informative holistic criteria. If a lending policy instead prioritizes **maximum overall accuracy (`0.7800`) and precision (`0.7500`)**, **Support Vector Machine (RBF)** and **Logistic Regression** are the top choices. |
