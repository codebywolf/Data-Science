# Healthcare Premium Prediction (Regression)

Streamlit app + ML pipeline to **predict annual healthcare insurance premium** from customer demographics, lifestyle, and medical history.

This project uses a **segmented modeling strategy by age**:
- **Young segment**: \(Age \(≤\) 25\) → model includes **`genetical_risk`**
- **Rest segment**: \(Age \(>\) 25\) → separate model + scaler

---

## Project structure

- **`app/main.py`**: Streamlit UI (form inputs + Predict button)
- **`app/predict.py`**: preprocessing + model routing + prediction
- **`app/artifacts/`**:
  - `model_young.joblib`, `scaler_young.joblib`
  - `model_rest.joblib`, `scaler_rest.joblib`
- **Notebooks (training / EDA)**:
  - `ml_premium_prediction.ipynb`: baseline end-to-end regression (EDA → feature engineering → model training/eval)
  - `dataset_split_by_age.ipynb`: splits the master dataset into age-based subsets
  - `ml_premium_prediction_young_with_gr.ipynb`: young-only model (adds `genetical_risk`)
  - `ml_premium_prediction_rest_with_gr.ipynb`: rest-only model (keeps compatible schema; genetic risk set to 0)
- **`datasets/`**:
  - `premiums.xlsx` (source)
  - age-split files like `premiums_young.xlsx`, `premiums_rest.xlsx` (+ variants used in notebooks)

---

## How prediction works (high level)

`app/predict.py` does:
- **Feature engineering**: computes `normalized_risk_score` from the selected *Medical History* (supports combined conditions like `"Diabetes & Thyroid"`).
- **One-hot / ordinal encoding**:
  - one-hot for fields like Gender/Region/Smoking/etc (base categories are implicit, like typical `drop_first=True`)
  - ordinal encoding for Insurance Plan: Bronze=1, Silver=2, Gold=3
- **Scaling**: applies the **age-appropriate MinMax scaler** (young vs rest) so inference matches training.
- **Model routing**: chooses `model_young` if Age \(≤\) 25 else `model_rest`.

Output is returned as an **integer premium**.

---

## Run the Streamlit app

### Prerequisites
- **Python 3.10+** (tested locally with 3.12)

### Install dependencies

If you don’t already have an environment:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
```

Install packages (minimum set used by app + notebooks):

```bash
pip install streamlit pandas scikit-learn joblib xgboost openpyxl
```

### Start the app

From the repo root:

```bash
streamlit run app/main.py
```

---

## Inputs used by the app

The UI collects:
- Age, Number of Dependants, Income (Lakhs)
- Genetical Risk
- Insurance Plan, Employment Status
- Gender, Marital Status, BMI Category
- Smoking Status, Region
- Medical History

---

## Training notes (where artifacts come from)

- Baseline modeling is in `ml_premium_prediction.ipynb`.
- To reduce error for younger customers, the pipeline introduces **model segmentation**:
  - young customers: `ml_premium_prediction_young_with_gr.ipynb`
  - older customers: `ml_premium_prediction_rest_with_gr.ipynb`
- The trained models and scalers are saved into `app/artifacts/` and loaded by the Streamlit app at runtime.

---

## Common issue

- **`FileNotFoundError: ./artifacts/...`**: run Streamlit from the repo root (`streamlit run app/main.py`) so relative artifact paths resolve correctly.
