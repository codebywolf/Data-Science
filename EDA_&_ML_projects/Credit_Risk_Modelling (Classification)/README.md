# 💳 Credit Card Risk Prediction

A machine learning project to predict credit card default using an XGBoost model and a Streamlit web interface.

## 🎯 Objective

This project aims to solve a critical business problem for financial institutions: accurately identifying customers who are likely to default on their credit obligations. By predicting default risk, the model helps in making informed lending decisions, thereby minimizing potential financial losses and promoting a healthier loan portfolio.

## 📊 Dataset Overview

The model is trained on a comprehensive dataset that combines customer demographics, loan application details, and credit bureau history.

-   **Key Features Used:**
    -   `credit_utilization_ratio`: Percentage of available credit the customer is using.
    -   `delinquent_ratio`: Ratio of months the customer was delinquent.
    -   `avg_dpd_per_delinquent`: Average days past due during delinquency.
    -   `loan_to_income`: Ratio of the loan amount to the customer's income.
    -   `age`, `residence_type`, `loan_purpose`, etc.
-   **Target Variable:** `default` (1 for Default, 0 for No Default).
-   **Class Imbalance:** The dataset is highly imbalanced with a **10.63:1** ratio of non-defaulters to defaulters, which was a key challenge addressed during modeling.

## 🏗️ Project Structure

```
.
├── app
│   ├── main.py               # Streamlit frontend application
│   └── prediction_helper.py  # Helper functions for prediction and scoring
├── artifacts
│   └── model_data.joblib     # Serialized model and preprocessing pipeline
├── datasets
│   ├── bureau_data.csv
│   ├── customers.csv
│   └── loans.csv
├── healthcare_premium_prediction.ipynb  # Jupyter Notebook for EDA and Model Training
└── README.md                 # This file
```

## ⚙️ ML Pipeline

The project follows a structured machine learning pipeline from raw data to a deployable model:

1.  **Data Loading & Merging:** Three separate CSV files (`customers`, `loans`, `bureau`) were loaded and merged into a single master DataFrame.
2.  **Data Cleaning:** Handled missing values (mode imputation for `residence_type`), removed outliers (e.g., processing fees >3% of loan amount), and corrected data entry errors (e.g., 'Personaal' -> 'Personal').
3.  **Feature Engineering:** Created powerful new features to capture complex risk behaviors:
    -   `loan_to_income`: Measures debt burden relative to income.
    -   `delinquent_ratio`: Normalizes delinquency history over the customer's loan lifetime.
    -   `avg_dpd_per_delinquent`: Measures the *severity* of delinquency, not just its occurrence.
4.  **Feature Selection:** Employed a two-step process to select the most predictive and independent features:
    -   **VIF Analysis:** Removed features with high Variance Inflation Factor (>5) to eliminate multicollinearity.
    -   **Information Value (IV):** Kept only features with IV > 0.02, ensuring all selected variables had meaningful predictive power.
5.  **Preprocessing:** Applied `MinMaxScaler` to scale numerical features and one-hot encoding for categorical features to prepare the data for the model.
6.  **Model Training:** Trained an XGBoost Classifier, which was chosen for its superior performance over baseline models like Logistic Regression and Random Forest.
7.  **Hyperparameter Tuning:** Utilized `RandomizedSearchCV` to find the optimal hyperparameters, with a specific focus on maximizing the recall of the 'default' class by tuning `scale_pos_weight`.
8.  **Prediction & Scoring:** The final model predicts the probability of default, which is then converted into a user-friendly credit score (300-900) and rating.

## 🧠 Model Details

The final model is an **XGBoost Classifier**. It was chosen for its ability to handle complex, non-linear relationships, its built-in regularization to prevent overfitting, and its robust performance on imbalanced datasets, which significantly outperformed other baseline models.

The model was fine-tuned to achieve the best balance between precision and recall, with the following optimal hyperparameters:

-   `subsample`: 0.8
-   `scale_pos_weight`: 8
-   `reg_lambda`: 0.5
-   `reg_alpha`: 0.1
-   `n_estimators`: 100
-   `min_child_weight`: 7
-   `max_depth`: 5
-   `learning_rate`: 0.01
-   `gamma`: 0.5
-   `colsample_bytree`: 0.9
-   `colsample_bylevel`: 0.6

## 📈 Model Performance

The model's performance was evaluated on an unseen test set, demonstrating excellent predictive power and reliability. The key business metric, **Recall** for the default class, was successfully optimized to 94%.

| Metric              | Score  |
| ------------------- | :----: |
| **AUC**             | 0.9842 |
| **KS Statistic**    | ~85    |
| **Accuracy**        |  93%   |
| **Precision (Default)** | 0.56   |
| **Recall (Default)**    | 0.94   |
| **F1-Score (Default)**  | 0.70   |

## 🔍 Key Insights

-   **Top Features:** The most influential feature is **`credit_utilization_ratio`** (importance = 0.38), confirming that how much credit a customer uses is the strongest predictor of default. This was followed by the engineered delinquency features.
-   **Data Leakage:** No data leakage was found. A check on `credit_utilization_ratio` showed a clear separation between the average for good customers (39.84%) and bad customers (81.52%), confirming the feature's predictive power is genuine.
-   **Multicollinearity:** VIF analysis was crucial. After removing redundant financial features, all remaining variables had a VIF score well below 5, with the top feature `credit_utilization_ratio` having a VIF of **2.94**.
-   **Overfitting:** The model shows no signs of overfitting. The gap between the Train AUC (0.9829) and Test AUC (0.9837) is a negligible **0.0008**, indicating excellent generalization.

## 💳 Credit Score Logic

The model's output probability is converted into an intuitive 300-900 credit score to make the risk assessment easy to understand.

1.  The model predicts the probability of default (`P(Default)`).
2.  The probability of non-default is calculated: `P(Non-Default) = 1 - P(Default)`.
3.  This is mapped to the score range using the formula:
    `Score = 300 + (P(Non-Default) * 600)`
4.  The final score is mapped to a rating:

| Score Range | Rating    |
| :---------: | :-------: |
|  750 - 900  | Excellent |
|  650 - 749  | Good      |
|  500 - 649  | Average   |
|  300 - 499  | Poor      |

## 🚀 How to Run

Follow these steps to set up and run the Streamlit web application locally.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/codebywolf/Data-Science.git
    cd "Data-Science/EDA_&_ML_projects/Credit_Risk_Modelling (Classification)"
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Streamlit application:**
    ```bash
    streamlit run app/main.py
    ```

## 📁 Artifacts

The `artifacts/model_data.joblib` file is a serialized dictionary containing the entire prediction pipeline, which includes:

-   **`model`**: The trained XGBoost classifier object.
-   **`scaler`**: The fitted `MinMaxScaler` object to scale new input data.
-   **`features`**: The exact list and order of feature names that the model expects.
-   **`col_to_scale`**: The list of columns that require scaling.

This single file makes the model easily portable and ensures that the exact same preprocessing steps are applied during prediction as were used during training, guaranteeing consistency.

## 🛠️ Tech Stack

-   **Python 3.9+**
-   **Pandas**: For data manipulation and analysis.
-   **Scikit-learn**: For data preprocessing, splitting, and model evaluation.
-   **XGBoost**: For the core classification model.
-   **Streamlit**: For building and serving the interactive web application.
-   **Joblib**: For model serialization and persistence.
-   **Seaborn & Matplotlib**: For data visualization.
