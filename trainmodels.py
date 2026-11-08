# ==========================================================
# MACHINE LEARNING ASSIGNMENT - 2
# M.Tech (Artificial Intelligence and Machine Learning)
# BITS PILANI
#
# Project: Comparison of Five Classification Models
# Dataset: Breast Cancer Wisconsin (Diagnostic)
# ==========================================================


# ==========================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ==========================================================

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Import Machine Learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Import evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ==========================================================
# STEP 2: SET PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "wdbc.data"

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)


# ==========================================================
# STEP 3: LOAD THE DATASET
# ==========================================================

print("\n==========================================")
print("       LOADING MACHINE LEARNING DATASET")
print("==========================================")

df = pd.read_csv(DATASET_PATH, header=None)

# Assign column names
df.columns = (
    ["id", "diagnosis"]
    + [f"feature_{i}" for i in range(1, 31)]
)

print("\nDataset Loaded Successfully!")

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nTotal Number of Records:", df.shape[0])

print("Total Input Features:", df.shape[1] - 2)


# ==========================================================
# STEP 4: DATA PREPROCESSING
# ==========================================================

print("\n==========================================")
print("           DATA PREPROCESSING")
print("==========================================")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert diagnosis into numerical labels
# Benign = 0
# Malignant = 1

df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

# Check target conversion
if df["diagnosis"].isnull().any():
    raise ValueError("Invalid diagnosis labels found.")

# Separate independent variables (features)
X = df.drop(columns=["id", "diagnosis"])

# Separate dependent variable (target)
y = df["diagnosis"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)

print("\nClass Distribution:")
print(y.value_counts())


# ==========================================================
# STEP 5: TRAINING AND TESTING DATA SPLIT
# ==========================================================

print("\n==========================================")
print("       TRAINING AND TESTING DATA")
print("==========================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)


# ==========================================================
# STEP 6: SAVE TEST DATASET
# ==========================================================

test_data = X_test.copy()

test_data["diagnosis"] = y_test

test_data.to_csv(
    BASE_DIR / "test_data.csv",
    index=False
)

print("\nTest dataset saved successfully!")
print("File Name: test_data.csv")


# ==========================================================
# STEP 7: DEFINE FIVE MACHINE LEARNING MODELS
# ==========================================================

print("\n==========================================")
print("       INITIALIZING ML CLASSIFIERS")
print("==========================================")

models = {

    # MODEL 1: LOGISTIC REGRESSION
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    ),

    # MODEL 2: DECISION TREE CLASSIFIER
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    # MODEL 3: K-NEAREST NEIGHBORS
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=5
        )
    ),

    # MODEL 4: NAIVE BAYES
    "Naive Bayes": GaussianNB(),

    # MODEL 5: SUPPORT VECTOR MACHINE
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            random_state=42
        )
    )
}

# Filenames for saving the trained models
model_filenames = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}


# ==========================================================
# STEP 8: TRAIN AND EVALUATE ALL FIVE MODELS
# ==========================================================

print("\n==========================================")
print("      TRAINING AND MODEL EVALUATION")
print("==========================================")

results = []

for model_name, model in models.items():

    print("\n------------------------------------------")
    print("Training Model:", model_name)
    print("------------------------------------------")

    # Train model using training dataset
    model.fit(X_train, y_train)

    print("Model Training Completed!")

    # Predict target values
    y_pred = model.predict(X_test)

    # Obtain probability estimates or decision scores
    # for AUC calculation
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(y_test, y_score)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Display metrics
    print("\nEvaluation Metrics:")

    print("Accuracy :", round(accuracy, 4))
    print("AUC Score:", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("MCC Score:", round(mcc, 4))

    # Store results in a list
    results.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # Display confusion matrix
    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print(cm)

    # Display classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"],
            zero_division=0
        )
    )

    # Save the trained model
    model_path = MODEL_DIR / model_filenames[model_name]

    joblib.dump(
        model,
        model_path
    )

    print("\nTrained Model Saved:", model_path.name)


# ==========================================================
# STEP 9: CREATE MODEL COMPARISON TABLE
# ==========================================================

print("\n==========================================")
print("       FINAL MODEL COMPARISON TABLE")
print("==========================================")

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================================
# STEP 10: SAVE MODEL COMPARISON RESULTS
# ==========================================================

comparison_path = BASE_DIR / "model_comparison.csv"

results_df.to_csv(
    comparison_path,
    index=False
)

print("\nModel Comparison Results Saved Successfully!")
print("File Name: model_comparison.csv")


# ==========================================================
# STEP 11: IDENTIFY BEST PERFORMING MODEL
# ==========================================================

# Select the model with the highest MCC score.
# MCC is used here as a balanced classification metric.

best_model_index = results_df["MCC"].idxmax()

best_model = results_df.loc[best_model_index]

print("\n==========================================")
print("            BEST PERFORMING MODEL")
print("==========================================")

print("\nModel Name:", best_model["ML Model Name"])

print("Accuracy:", round(best_model["Accuracy"], 4))

print("AUC Score:", round(best_model["AUC"], 4))

print("F1 Score:", round(best_model["F1"], 4))

print("MCC Score:", round(best_model["MCC"], 4))


# ==========================================================
# STEP 12: FINAL EXECUTION SUMMARY
# ==========================================================

print("\n==========================================")
print("           EXECUTION COMPLETED")
print("==========================================")

print("\nAll Five ML Models Trained Successfully!")

print("\nModels Implemented:")
print("1. Logistic Regression")
print("2. Decision Tree Classifier")
print("3. K-Nearest Neighbors")
print("4. Gaussian Naive Bayes")
print("5. Support Vector Machine")

print("\nEvaluation Metrics Calculated:")
print("1. Accuracy")
print("2. AUC Score")
print("3. Precision")
print("4. Recall")
print("5. F1 Score")
print("6. MCC Score")

print("\nGenerated Files:")
print("1. test_data.csv")
print("2. model_comparison.csv")
print("3. Five trained models inside model/ folder")

print("\nAssignment Model Training Phase Completed!")