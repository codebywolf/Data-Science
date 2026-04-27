# Import necessary libraries
from joblib import load  # For loading the saved model artifacts
import pandas as pd      # For data manipulation
import os                # For interacting with the operating system, e.g., to build file paths

# --- Load Model Artifacts ---

# 1. Get the absolute path of the directory where this script ('prediction_helper.py') is located.
app_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Navigate one level up from the 'app' directory to get to the root project directory.
root_dir = os.path.dirname(app_dir)

# 3. Construct the full path to the model artifacts file.
MODEL_PATH = os.path.join(root_dir, 'artifacts', 'model_data.joblib')

# Debugging: Print the constructed path to ensure it's correct.
print(f"DEBUG: Looking for model at: {MODEL_PATH}")

# Load the dictionary of model artifacts from the .joblib file.
model_data = load(MODEL_PATH)
# Unpack the artifacts from the dictionary into individual variables.
model = model_data['model']              # The trained XGBoost model
scaler = model_data['scaler']            # The fitted MinMaxScaler
features = model_data['features']        # The list of feature names the model expects
cols_to_scale = model_data['col_to_scale'] # The list of columns that require scaling

def prepare_df(age, income, loan_amount, loan_tenure_months,
            avg_dpd_per_delinquent, delinquent_ratio,
            credit_utilization_ratio, number_of_open_accounts,
            residence_type, loan_purpose,
            loan_type):
    """
    Prepares a DataFrame from user inputs, performs one-hot encoding, scaling,
    and feature selection to match the model's training format.
    """

    # Create a dictionary from the input data, including engineered features and one-hot encoding.
    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': number_of_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0, # Engineered feature
        'delinquent_ratio': delinquent_ratio,
        'avg_dpd_per_delinquent': avg_dpd_per_delinquent,
        # Manual one-hot encoding for categorical features
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
    }

    # Convert the dictionary into a single-row pandas DataFrame.
    df = pd.DataFrame([input_data])
    # Apply the pre-fitted scaler to the specified columns.
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    # Clip values to be between 0 and 1 to handle any potential scaling issues with unseen data.
    df[cols_to_scale] = df[cols_to_scale].clip(0, 1)
    # Ensure the DataFrame columns are in the exact same order as during model training.
    df = df[features]

    return df

def calculate_credit_score(input_df, model, base_score=300, scale_length=600):
    """
    Calculates the credit score and rating based on the model's default probability.
    """

    # Get the predicted probability of the positive class (default=1).
    # predict_proba returns probabilities for [class_0, class_1].
    default_probability = model.predict_proba(input_df)[0][1]
    # The score is based on the probability of NOT defaulting.
    non_default_probability = 1 - default_probability
    # Ensure the probability is within the valid range [0, 1].
    non_default_probability = max(0.0, min(1.0, non_default_probability))

    # Convert the non-default probability into a credit score on a scale (e.g., 300-900).
    credit_score = base_score + (non_default_probability * scale_length)
    final_score = int(credit_score)

    # Define a function to map the credit score to a qualitative rating.
    def get_rating(score):
        if 300 <= score < 500: return "Poor"
        elif 500 <= score < 650: return "Average"
        elif 650 <= score < 750: return "Good"
        elif 750 <= score <= 900: return "Excellent"
        return "Invalid Score"

    # Get the rating for the calculated score.
    rating = get_rating(final_score)

    return default_probability, final_score, rating

def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquent, delinquent_ratio,
            credit_utilization_ratio, number_of_open_accounts, residence_type, loan_purpose, loan_type):
    """
    Main prediction function that orchestrates the data preparation and scoring.
    """

    # 1. Prepare the input data into a format the model understands.
    input_df = prepare_df(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquent, delinquent_ratio,
            credit_utilization_ratio, number_of_open_accounts, residence_type, loan_purpose, loan_type)

    # 2. Use the prepared data to calculate the risk probability, score, and rating.
    probability, credit_score, rating = calculate_credit_score(input_df, model)

    # 3. Return the final results.
    return probability, credit_score, rating
