# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import predictor # Your module with prediction logic and artifact loading
import time # For simulating some processing or giving feedback

# --- Page Configuration (Optional) ---
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="wide"
)

# --- Load Models and Preprocessors ---
# This should ideally run only once. Streamlit has caching mechanisms.
@st.cache_resource # Use cache_resource for objects like models, LEs, scalers
def load_all_artifacts():
    try:
        predictor.load_artifacts() # Calls the function in your predictor.py
        return True # Indicate success
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return False # Indicate failure

artifacts_loaded = load_all_artifacts()

# --- App Title and Description ---
st.title("💳 GNN Embeddings + XGBoost Fraud Detection")
st.markdown("""
This application uses a hybrid model to predict the likelihood of a transaction being fraudulent.
Input the transaction details below to get a prediction.
""")

if not artifacts_loaded:
    st.error("Model artifacts could not be loaded. The app cannot make predictions. Please check server logs or artifact paths.")
    st.stop() # Stop further execution of the app

# --- Input Fields for Transaction Data ---
st.sidebar.header("Transaction Input")

# Using a form for better UX if there are many inputs
with st.sidebar.form(key='transaction_form'):
    # Match these keys with what your `preprocess_transaction_for_gnn` expects
    # and the original CSV column names for user input clarity.
    user_id = st.text_input("User ID", value="test_user_001") # Example default
    card_id = st.text_input("Card Number (last 4 digits or ID)", value="1234")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2050, value=2023, step=1)
    with col2:
        month = st.number_input("Month", min_value=1, max_value=12, value=10, step=1)
    with col3:
        day = st.number_input("Day", min_value=1, max_value=31, value=28, step=1)

    transaction_time = st.text_input("Time (HH:MM:SS)", value="14:35:00")
    amount_str = st.text_input("Amount (e.g., $123.45 or 123.45)", value="$55.60")
    
    use_chip = st.selectbox("Transaction Type (Use Chip)", ["Chip Transaction", "Online Transaction", "Swipe Transaction"], index=0)
    
    # For merchant details, you might need to map names to IDs if your LE expects IDs
    # For simplicity, let's assume text input for now
    merchant_name_input = st.text_input("Merchant Name/ID", value="Merchant_XYZ_789") # Use a known one if testing LEs
    merchant_city = st.text_input("Merchant City", value="NEW YORK")
    merchant_state = st.text_input("Merchant State", value="NY")
    zip_code = st.text_input("Zip Code", value="10001")
    mcc = st.text_input("MCC (Merchant Category Code)", value="5411") # Example: Grocery
    
    errors = st.selectbox("Errors?", ["No Error", "Technical Glitch", "Bad PIN", "Insufficient Balance"], index=0) # Match values your LE expects

    # Every form must have a submit button.
    submit_button = st.form_submit_button(label='Detect Fraud')

# --- Prediction Logic ---
if submit_button:
    if not all([user_id, card_id, transaction_time, amount_str, use_chip, merchant_name_input, merchant_city, mcc, errors]):
        st.error("Please fill in all required fields.")
    else:
        with st.spinner('Analyzing transaction...'):
            # Prepare input dictionary for your predictor
            # Ensure keys match what preprocess_transaction_for_gnn expects
            transaction_data_dict = {
                'User': user_id,
                'Card': card_id,
                'Year': year,
                'Month': month,
                'Day': day,
                'Time': transaction_time, # Your preprocess function handles parsing this with DateString
                'Amount': amount_str, # Your preprocess function handles stripping '$'
                'Use Chip': use_chip,
                'Merchant Name': merchant_name_input, # Your preprocess maps this to merchant_idx
                'Merchant City': merchant_city,
                'Merchant State': merchant_state,
                'Zip': zip_code,
                'MCC': mcc,
                'Errors?': errors,
                'TransactionID': f"streamlit_tx_{time.time()}" # Generate a unique ID
            }

            try:
                prediction_result = predictor.predict_fraud(transaction_data_dict)

                st.subheader("Prediction Result")
                if "error" in prediction_result:
                    st.error(f"Prediction Error: {prediction_result['error']}")
                    if "details" in prediction_result:
                        st.caption(f"Details: {prediction_result['details']}")
                else:
                    fraud_score = prediction_result.get("fraud_score", 0.0)
                    is_fraud = prediction_result.get("is_fraud_prediction", 0)
                    threshold = prediction_result.get("threshold_used", 0.5)

                    if is_fraud == 1:
                        st.error(f"🚨 Transaction Flagged as POTENTIALLY FRAUDULENT (Score: {fraud_score:.4f})")
                    else:
                        st.success(f"✅ Transaction Predicted as LIKELY NOT FRAUDULENT (Score: {fraud_score:.4f})")
                    
                    st.markdown(f"**Fraud Score:** `{fraud_score:.4f}` (Threshold for flagging: `{threshold:.4f}`)")
                    
                    # Display some input data for confirmation
                    st.markdown("---")
                    st.markdown("**Input Summary:**")
                    st.json({k: v for k, v in transaction_data_dict.items() if k != 'TransactionID'}, expanded=False)

            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")
                traceback.print_exc() # This will print to the console where Streamlit is running

st.sidebar.markdown("---")
st.sidebar.info("This app uses a GNN for feature embeddings and XGBoost for classification.")

