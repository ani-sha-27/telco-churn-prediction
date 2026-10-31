# Telco Customer Churn Prediction

I built this to predict which telecom customers are likely to leave. It uses the Telco Customer Churn dataset from Kaggle (7,043 customers, about 26.5% of them churned). The model is a Logistic Regression, and I wrapped it in a small FastAPI app so you can send it one customer's details and get back a churn probability.

## What's in the repo

- `Churn_Prediction_.ipynb` - the whole notebook: cleaning, EDA, models, tuning, saving
- `main.py` - the API
- `churn_model.pkl`, `scaler.pkl`, `threshold.pkl` - the saved model, scaler and cutoff
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` - the data
- `requirements.txt`

## How I built it

**Cleaning.** `TotalCharges` has 11 blank values. All 11 are customers with 0 months of tenure, so they just hadn't been billed yet. I filled them with 0. I dropped `customerID`, turned the Yes/No columns into 1/0, and one-hot encoded `InternetService`, `Contract` and `PaymentMethod`.

**What the EDA showed.** Month-to-month contracts churn at about 43%, against under 3% for two-year contracts. Fiber optic customers churn a lot more than DSL ones (about 42% vs 19%), and so do people paying by electronic check. Customers with online security or tech support leave much less. Gender makes almost no difference.

**Models.** I tried Logistic Regression, Random Forest and XGBoost. Only about 1 in 4 customers churn, so I used `class_weight='balanced'` for Logistic Regression and `scale_pos_weight` for XGBoost. On the test set the ROC-AUC was 0.862 for Logistic Regression, 0.857 for XGBoost and 0.846 for Random Forest. I went with Logistic Regression because it scored best and because its coefficients are easy to explain. I tuned `C` with 5-fold cross-validation and `C=2` won.

**Threshold.** I used 0.40 instead of the default 0.5. Missing a customer who is about to leave usually costs more than sending a retention offer to someone who would have stayed anyway. At 0.40 the model catches about 90% of the churners in the test set (recall 0.895), but only 47% of the customers it flags actually leave (precision 0.466).

## Running the API

```
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000/docs`, click `POST /predict` and try it out. The input uses the same values as the original CSV:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.7,
  "TotalCharges": 151.65
}
```

And you get back:

```json
{
  "churn_probability": 0.8515,
  "churn_prediction": 1,
  "threshold": 0.4,
  "result": "Likely to churn"
}
```

The API rejects combinations that don't make sense, like add-on services marked "Yes" when the customer has no internet.

## Things to know

- I chose the 0.40 threshold by looking at results on the test set, so the recall number above is probably a little optimistic. Picking it from cross-validated training predictions would be the cleaner way to do it.
- The pickle files were saved with scikit-learn 1.6.1. Newer versions can load them, but you'll see a version warning when the API starts.
- This is a public sample dataset, not real company data.
