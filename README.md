# Machine Learning Assignment 2 — Classification Model Comparison

**Programme:** M.Tech (Artificial Intelligence and Machine Learning)  
**Course:** Machine Learning  
**Institution:** BITS Pilani — Work Integrated Learning Programmes  
**Student name:** VAGESH GAUTAM  
**Roll number:** 2026AK05202

## 1. Problem Statement

The objective of this project is to build, evaluate, and compare five supervised machine-learning algorithms for a binary classification problem. The chosen task is to classify diagnostic observations in the **Breast Cancer Wisconsin (Diagnostic)** dataset as benign or malignant using numerical measurements derived from cell nuclei.

The project follows a reproducible workflow: inspect and preprocess the dataset, separate training and test records, fit all five classifiers on the same training set, assess them on the same held-out test set, save trained models, and create an interactive Streamlit application that evaluates the saved models on an uploaded test CSV **without retraining**.

The five required algorithms are Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).

## 2. Dataset Description

| Item | Description |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Original source | UCI Machine Learning Repository |
| Dataset page | <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> |
| Problem type | Binary classification |
| Total instances | 569 |
| Input features | 30 numeric features |
| Additional original columns | Sample ID and diagnosis |
| Target variable | `diagnosis` |
| Target mapping | Benign (`B`) = `0`; Malignant (`M`) = `1` |
| Training/test split | 80% / 20%, using `random_state=42` and stratification |
| Training observations | 455 |
| Held-out test observations | 114 |

### Data preparation

1. Read the original `wdbc.data` file without a header and assign the column names `id`, `diagnosis`, and `feature_1` through `feature_30`.
2. Inspect the data, its dimensions, and missing-value counts.
3. Convert diagnosis labels from `B`/`M` to `0`/`1`.
4. Exclude `id` and `diagnosis` from the model input features.
5. Create one shared, stratified 80:20 training/test split for all five classifiers.
6. Apply `StandardScaler` **within the trained pipeline** for Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes use the numeric features without this scaling step.
7. Export the untouched held-out test partition, including the target column, to `test_data.csv` for application evaluation.

The models are fitted on training data only. All reported evaluation results must come from the held-out test data.

## 3. GitHub Repository Link

**Public GitHub repository:** https://github.com/VageshGautam/-ML-Assignment-2-2026ak05202-Sem1-bitspilani-wilp

**Live Streamlit application:** https://yvgzag8ecm59jlplc4yhii.streamlit.app/

The GitHub repository contains the source code, `requirements.txt`, this `README.md`, `test_data.csv`, and five trained model files in the `model/` directory.

## 4. Models Used and Evaluation Metrics

| Model | Configuration in `train_models.py` |
|---|---|
| Logistic Regression | Standard scaling followed by `LogisticRegression(max_iter=1000, random_state=42)` |
| Decision Tree | `DecisionTreeClassifier(random_state=42)` |
| K-Nearest Neighbors | Standard scaling followed by `KNeighborsClassifier(n_neighbors=5)` |
| Gaussian Naive Bayes | `GaussianNB()` |
| Support Vector Machine | Standard scaling followed by `SVC(kernel="rbf", random_state=42)` |

### Evaluation methodology

Each model is evaluated on the **same test split** using the six required metrics:

- **Accuracy:** Proportion of correctly classified test records.
- **AUC (ROC-AUC):** Ability to rank positive observations ahead of negative observations across classification thresholds. Computed using predicted probabilities where available, or the SVM decision score.
- **Precision:** Proportion of predicted malignant cases that are actually malignant.
- **Recall:** Proportion of actual malignant cases detected by the classifier.
- **F1 Score:** Harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Balanced classification metric that accounts for true and false positives and negatives.

### Model comparison table
ML Model Name,Accuracy,AUC,Precision,Recall,F1,MCC
Logistic Regression,0.9649122807017544,0.996031746031746,0.975,0.9285714285714286,0.9512195121951219,0.9245181185940877
Decision Tree,0.9298245614035088,0.9246031746031745,0.9047619047619048,0.9047619047619048,0.9047619047619048,0.8492063492063492
KNN,0.956140350877193,0.982308201058201,0.9743589743589743,0.9047619047619048,0.9382716049382716,0.9058238738943076
Naive Bayes,0.9385964912280702,0.9933862433862434,1.0,0.8333333333333334,0.9090909090909091,0.8714893406611902
SVM,0.9736842105263158,0.9947089947089947,1.0,0.9285714285714286,0.9629629629629629,0.9441549509633318

**Source of results:** `model_comparison.csv`, generated by the training script from the held-out test partition.

## 5. Observations on Model Performance
------------------------------------------
Training Model: Logistic Regression
------------------------------------------
Model Training Completed!

Evaluation Metrics:
Accuracy : 0.9649
AUC Score: 0.996
Precision: 0.975
Recall   : 0.9286
F1 Score : 0.9512
MCC Score: 0.9245

Confusion Matrix:
[[71  1]
 [ 3 39]]

Classification Report:
              precision    recall  f1-score   support

      Benign       0.96      0.99      0.97        72
   Malignant       0.97      0.93      0.95        42

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114

Trained Model Saved: logistic_regression.joblib

------------------------------------------
Training Model: Decision Tree
------------------------------------------
Model Training Completed!

Evaluation Metrics:
Accuracy : 0.9298
AUC Score: 0.9246
Precision: 0.9048
Recall   : 0.9048
F1 Score : 0.9048
MCC Score: 0.8492

Confusion Matrix:
[[68  4]
 [ 4 38]]

Classification Report:
              precision    recall  f1-score   support

      Benign       0.94      0.94      0.94        72
   Malignant       0.90      0.90      0.90        42

    accuracy                           0.93       114
   macro avg       0.92      0.92      0.92       114
weighted avg       0.93      0.93      0.93       114


Trained Model Saved: decision_tree.joblib

------------------------------------------

------------------------------------------
Training Model: KNN
------------------------------------------
Model Training Completed!

Evaluation Metrics:
Accuracy : 0.9561
AUC Score: 0.9823
Precision: 0.9744
Recall   : 0.9048
F1 Score : 0.9383
MCC Score: 0.9058

Confusion Matrix:
[[71  1]
 [ 4 38]]

Classification Report:
              precision    recall  f1-score   support

      Benign       0.95      0.99      0.97        72
   Malignant       0.97      0.90      0.94        42

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

Trained Model Saved: knn.joblib

------------------------------------------
Training Model: Naive Bayes
------------------------------------------
Model Training Completed!

Evaluation Metrics:
Accuracy : 0.9386
AUC Score: 0.9934
Precision: 1.0
Recall   : 0.8333
F1 Score : 0.9091
MCC Score: 0.8715

Confusion Matrix:
[[72  0]
 [ 7 35]]

Classification Report:
              precision    recall  f1-score   support

      Benign       0.91      1.00      0.95        72
   Malignant       1.00      0.83      0.91        42

    accuracy                           0.94       114
   macro avg       0.96      0.92      0.93       114
weighted avg       0.94      0.94      0.94       114

Trained Model Saved: naive_bayes.joblib

------------------------------------------
Training Model: SVM
------------------------------------------
Model Training Completed!

Evaluation Metrics:
Accuracy : 0.9737
AUC Score: 0.9947
Precision: 1.0
Recall   : 0.9286
F1 Score : 0.963
MCC Score: 0.9442

Confusion Matrix:
[[72  0]
 [ 3 39]]

Classification Report:
              precision    recall  f1-score   support

      Benign       0.96      1.00      0.98        72
   Malignant       1.00      0.93      0.96        42

    accuracy                           0.97       114
   macro avg       0.98      0.96      0.97       114
weighted avg       0.97      0.97      0.97       114

Trained Model Saved: svm.joblib

==========================================
       FINAL MODEL COMPARISON TABLE
==========================================
      ML Model Name  Accuracy    AUC  Precision  Recall     F1    MCC
Logistic Regression    0.9649 0.9960     0.9750  0.9286 0.9512 0.9245
      Decision Tree    0.9298 0.9246     0.9048  0.9048 0.9048 0.8492
                KNN    0.9561 0.9823     0.9744  0.9048 0.9383 0.9058
        Naive Bayes    0.9386 0.9934     1.0000  0.8333 0.9091 0.8715
                SVM    0.9737 0.9947     1.0000  0.9286 0.9630 0.9442

Model Comparison Results Saved Successfully!
File Name: model_comparison.csv

==========================================
            BEST PERFORMING MODEL
==========================================

Model Name: SVM
Accuracy: 0.9737
AUC Score: 0.9947
F1 Score: 0.963
MCC Score: 0.9442

## 6. Project Structure

```text
ML_ASSIGNMENT_2/
├── app.py
├── train_models.py
├── dataset_analysis.py
├── requirements.txt
├── README.md
├── wdbc.data
├── test_data.csv
├── model_comparison.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

`dataset_analysis.py` is used for initial dataset exploration and train/test preparation. `train_models.py` fits and evaluates all five algorithms, creates `test_data.csv` and `model_comparison.csv`, and saves the trained models. `app.py` provides the interactive evaluator.

## 7. How to Run the Project Locally

### Requirements

- Python installed on your computer
- Packages listed in `requirements.txt`
- `wdbc.data` available in the project root

### Install dependencies

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

### Train and save the models

```bash
python train_models.py
```

This command creates `test_data.csv`, `model_comparison.csv`, and five `.joblib` files in `model/`.

### Launch the interactive app

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal (normally `http://localhost:8501`).

## 8. Streamlit Application Features

The application provides:

1. **Test CSV upload:** Upload the `test_data.csv` generated by the training script. The file must contain `feature_1` to `feature_30` and `diagnosis`.
2. **Model selection:** Choose Logistic Regression, Decision Tree, KNN, Naive Bayes, or SVM from a dropdown.
3. **Six evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 Score, and MCC for the selected model.
4. **Confusion matrix:** Visual display of actual versus predicted classes.
5. **Classification report:** Class-level precision, recall, and F1 scores.
6. **All-model comparison:** Comparison of all five saved models on the uploaded test data, with an option to download results as CSV.

**No retraining in the app:** The app loads model files with `joblib`, checks the uploaded columns, and evaluates predictions. Training occurs only in `train_models.py`.

## 9. Limitations and Responsible Use

- The reported metrics describe a single held-out train/test split; results may vary with a different split or model settings.
- The comparison is intended for academic demonstration, not clinical decision-making.
- Model predictions are not a medical diagnosis and must not replace evaluation by qualified healthcare professionals.

## 11. Reference

UCI Machine Learning Repository, *Breast Cancer Wisconsin (Diagnostic)*: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
