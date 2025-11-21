import streamlit as st
import pandas as pd
import joblib, json, os

st.set_page_config(page_title='NATA Supermarket Spend Prediction App', layout='centered')
st.title('NATA Supermarket')

MODEL_FILE = 'rf_model.pkl'
FEATURE_FILE = 'feature_columns.json'

model, feature_cols = None, None
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    st.success('✅ Model loaded successfully')
else:
    st.error('❌ Model file not found')

if os.path.exists(FEATURE_FILE):
    with open(FEATURE_FILE, 'r') as f:
        feature_cols = json.load(f)
else:
    st.error('❌ Feature list not found')

st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    Year_Birth = st.number_input("Year Birth", min_value=1900, max_value=2025, value=1980)
    Education = st.selectbox("Education", ["2n Cycle", "Basic", "Graduation", "Master", "PhD"])
    Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widow", "Alone", "YOLO", "Together", "Absurd"])
    Income = st.number_input("Income", min_value=0, value=30000)
    Kidhome = st.number_input("Kidhome", min_value=0, max_value=500, value=0)
    Teenhome = st.number_input("Teenhome", min_value=0, max_value=500, value=0)
    Recency = st.number_input("Recency (days)", min_value=0, max_value=1000000, value=10)

with col2:
    NumDealsPurchases = st.number_input("Num Deals Purchases", min_value=0, max_value=500, value=1)
    NumWebPurchases = st.number_input("Num Web Purchases", min_value=0, max_value=500, value=2)
    NumCatalogPurchases = st.number_input("Num Catalog Purchases", min_value=0, max_value=500, value=0)
    NumStorePurchases = st.number_input("Num Store Purchases", min_value=0, max_value=500, value=3)
    NumWebVisitsMonth = st.number_input("Num Web Visits Month", min_value=0, max_value=500, value=5)
    AcceptedCmp3 = st.selectbox("Accepted Campaign 3", [0, 1])
    AcceptedCmp4 = st.selectbox("Accepted Campaign 4", [0, 1])
    AcceptedCmp5 = st.selectbox("Accepted Campaign 5", [0, 1])
    AcceptedCmp1 = st.selectbox("Accepted Campaign 1", [0, 1])
    AcceptedCmp2 = st.selectbox("Accepted Campaign 2", [0, 1])
    Complain = st.selectbox("Complained?", [0, 1])
    Response = st.selectbox("Response", [0, 1])

input_df = pd.DataFrame([{
    'Year_Birth': Year_Birth,
    'Education': Education,
    'Marital_Status': Marital_Status,
    'Income': Income,
    'Kidhome': Kidhome,
    'Teenhome': Teenhome,
    'Recency': Recency,
    'NumDealsPurchases': NumDealsPurchases,
    'NumWebPurchases': NumWebPurchases,
    'NumCatalogPurchases': NumCatalogPurchases,
    'NumStorePurchases': NumStorePurchases,
    'NumWebVisitsMonth': NumWebVisitsMonth,
    'AcceptedCmp3': AcceptedCmp3,
    'AcceptedCmp4': AcceptedCmp4,
    'AcceptedCmp5': AcceptedCmp5,
    'AcceptedCmp1': AcceptedCmp1,
    'AcceptedCmp2': AcceptedCmp2,
    'Complain': Complain,
    'Response': Response
}])

# required feature order (your model input)
col = ['Income',
 'Kidhome',
 'Teenhome',
 'Recency',
 'Response',
 'NumDealsPurchases',
 'NumWebPurchases',
 'NumCatalogPurchases',
 'NumStorePurchases',
 'NumWebVisitsMonth',
 'AcceptedCmp3',
 'AcceptedCmp4',
 'AcceptedCmp5',
 'AcceptedCmp1',
 'AcceptedCmp2',
 'Complain',
 'Education_Basic',
 'Education_Graduation',
 'Education_Master',
 'Education_PhD',
 'Marital_Status_Alone',
 'Marital_Status_Divorced',
 'Marital_Status_Married',
 'Marital_Status_Single',
 'Marital_Status_Together',
 'Marital_Status_Widow',
 'Marital_Status_YOLO']

# When user clicks Predict
if st.button('Predict Total Spend'):
    if model is None or feature_cols is None:
        st.error('Model or feature list not loaded.')
    else:
        # Build a single-row dict from the Streamlit inputs
        row = {
            'Income': Income,
            'Kidhome': Kidhome,
            'Teenhome': Teenhome,
            'Recency': Recency,
            'Response': Response,
            'NumDealsPurchases': NumDealsPurchases,
            'NumWebPurchases': NumWebPurchases,
            'NumCatalogPurchases': NumCatalogPurchases,
            'NumStorePurchases': NumStorePurchases,
            'NumWebVisitsMonth': NumWebVisitsMonth,
            'AcceptedCmp3': AcceptedCmp3,
            'AcceptedCmp4': AcceptedCmp4,
            'AcceptedCmp5': AcceptedCmp5,
            'AcceptedCmp1': AcceptedCmp1,
            'AcceptedCmp2': AcceptedCmp2,
            'Complain': Complain
        }

        # Initialize one-hot columns with 0
        for c in col:
            if c not in row:
                row[c] = 0

        # Map Education -> one-hot columns used by model
        # Note: your selectbox contains "2n Cycle" which is not one of the model columns;
        # for such categories we keep all Education_* = 0 (same as unseen category).
        if Education == 'Basic':
            row['Education_Basic'] = 1
        elif Education == 'Graduation':
            row['Education_Graduation'] = 1
        elif Education == 'Master':
            row['Education_Master'] = 1
        elif Education == 'PhD':
            row['Education_PhD'] = 1
        # else (e.g. '2n Cycle' or any other) => leave all Education_* = 0

        # Map Marital_Status -> one-hot columns used by model
        # Your selectbox includes 'Absurd' — that will map to all zeros (unknown category)
        ms = Marital_Status
        if ms == 'Alone':
            row['Marital_Status_Alone'] = 1
        elif ms == 'Divorced':
            row['Marital_Status_Divorced'] = 1
        elif ms == 'Married':
            row['Marital_Status_Married'] = 1
        elif ms == 'Single':
            row['Marital_Status_Single'] = 1
        elif ms == 'Together':
            row['Marital_Status_Together'] = 1
        elif ms == 'Widow':
            row['Marital_Status_Widow'] = 1
        elif ms == 'YOLO':
            row['Marital_Status_YOLO'] = 1
        # else ('Absurd' etc.) => leave all Marital_Status_* = 0

        # Convert to DataFrame and ensure column order matches `col`
        X_pred = pd.DataFrame([row])[col]  # reorders and keeps only required cols

        # cast numeric types to avoid dtype issues
        X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)

        try:
            pred = model.predict(X_pred)[0]
            st.success(f'Predicted Total Spending: {pred:.2f}')
            with st.expander("Prepared features used for prediction"):
                st.write(X_pred.T)
        except Exception as e:
            st.error(f'Prediction failed: {e}')
