# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import predictor # Your module
import time
import traceback

# --- Page Configuration ---
st.set_page_config(page_title="Fraud Detection App", page_icon="🛡️", layout="wide")

# --- Load Artifacts ---
@st.cache_resource # Cache the loading process
def load_all_artifacts_streamlit():
    st.info("Loading model artifacts... This may take a moment on first run.")
    try:
        success = predictor.load_artifacts()
        if success:
            st.success("Model artifacts loaded successfully!")
        else:
            st.error("Failed to load some essential model artifacts. Predictions may be unreliable or fail.")
        return success
    except Exception as e:
        st.error(f"Critical error during artifact loading: {e}")
        traceback.print_exc()
        return False

artifacts_loaded_successfully = load_all_artifacts_streamlit()

# --- App UI ---
st.title("💳 Hybrid GNN + XGBoost Fraud Detection")
st.markdown("Enter transaction details to predict if it's fraudulent.")

if not artifacts_loaded_successfully:
    st.error("Application cannot proceed: Essential model artifacts failed to load. Please check the server console.")
    st.stop()

st.sidebar.header("Transaction Input")
with st.sidebar.form(key='transaction_input_form'):
    # Input fields (as in your previous Streamlit example)
    user_id = st.text_input("User ID", value="0") # Default to a known user if possible
    card_id = st.text_input("Card ID/Number", value="card_A_for_user_0") # Default to a known card for user 0

    # Use actual known values if available from loaded LabelEncoders for better demo
    if predictor.user_le_loaded and len(predictor.user_le_loaded.classes_) > 0:
        default_user_streamlit = predictor.user_le_loaded.classes_[0]
        user_id = st.text_input("User ID (e.g., from training data)", value=default_user_streamlit)
        if predictor.user_to_cards_map_loaded and default_user_streamlit in predictor.user_to_cards_map_loaded and predictor.user_to_cards_map_loaded[default_user_streamlit]:
            default_card_streamlit = list(predictor.user_to_cards_map_loaded[default_user_streamlit])[0]
            card_id = st.text_input("Card ID/Number (known for above user)", value=default_card_streamlit)
    
    # Date and Time
    col1_st, col2_st, col3_st = st.columns(3)
    with col1_st: year = st.number_input("Year", 2000, 2050, pd.Timestamp.now().year)
    with col2_st: month = st.number_input("Month", 1, 12, pd.Timestamp.now().month)
    with col3_st: day = st.number_input("Day", 1, 31, pd.Timestamp.now().day)
    transaction_time_str = st.text_input("Time (HH:MM:SS)", value=pd.Timestamp.now().strftime("%H:%M:%S"))
    
    amount_str = st.text_input("Amount (e.g., $123.45 or 123.45)", value="$75.50")
    
    use_chip_options = ["Chip Transaction", "Online Transaction", "Swipe Transaction"] # Ensure these match your LE
    use_chip = st.selectbox("Transaction Type", use_chip_options, index=0)
    
    # For merchant details, provide known examples if possible
    default_merchant_name = predictor.merchant_le_loaded.classes_[0] if predictor.merchant_le_loaded and len(predictor.merchant_le_loaded.classes_) > 0 else "12345"
    merchant_name_input = st.text_input("Merchant Name/ID", value=default_merchant_name)
    
    default_mcc_val = predictor.cat_le_dict_loaded['MCC'].classes_[0] if predictor.cat_le_dict_loaded and 'MCC' in predictor.cat_le_dict_loaded and len(predictor.cat_le_dict_loaded['MCC'].classes_)>0 else "5411"
    mcc = st.text_input("MCC", value=default_mcc_val)

    merchant_city = st.text_input("Merchant City", value="NEW YORK") # Example
    merchant_state = st.text_input("Merchant State", value="NY") # Example
    zip_code = st.text_input("Zip Code", value="10001") # Example

    errors_options = ["No Error", "Technical Glitch", "Bad PIN", "Insufficient Balance", "Bad CVV", "Bad Card Number", "Bad Expiration"]
    errors = st.selectbox("Errors?", errors_options, index=0)
    
    submit_button = st.form_submit_button(label='🔍 Predict Fraud')

if submit_button:
    with st.spinner('🧠 Analyzing transaction...'):
        transaction_data_dict = {
            'User': user_id, 'Card': card_id, 'Year': year, 'Month': month, 'Day': day,
            'Time': transaction_time_str, 'Amount': amount_str, 'Use Chip': use_chip,
            'Merchant Name': merchant_name_input, 'Merchant City': merchant_city,
            'Merchant State': merchant_state, 'Zip': zip_code, 'MCC': mcc, 'Errors?': errors,
            'TransactionID': f"st_app_tx_{int(time.time())}"
        }
        
        st.markdown("---")
        st.markdown("**Input Received by Predictor:**")
        st.json({k: v for k,v in transaction_data_dict.items() if k != 'TransactionID'}) # Show what's sent

        prediction_output, status_code = predictor.predict_fraud(transaction_data_dict)
        
        st.subheader("Prediction Result")
        if "error" in prediction_output:
            st.error(f"🚫 Error (Status {status_code}): {prediction_output['error']}")
            if "details" in prediction_output: st.caption(f"Details: {prediction_output['details']}")
            if "user_provided" in prediction_output: # For card validation error
                st.warning(f"User: {prediction_output['user_provided']}, Card: {prediction_output['card_provided']}")
        else:
            fraud_score = prediction_output.get("fraud_score", 0.0)
            is_fraud = prediction_output.get("is_fraud_prediction", 0)
            threshold = prediction_output.get("threshold_used", 0.5)
            card_status = prediction_output.get("card_validation_status", "Unknown")

            if card_status != "Validated":
                st.warning(f"Card Validation Status: {card_status}")

            if is_fraud == 1:
                st.error(f"🚨 POTENTIALLY FRAUDULENT (Score: {fraud_score:.4f})")
            else:
                st.success(f"✅ LIKELY NOT FRAUDULENT (Score: {fraud_score:.4f})")
            st.metric(label="Fraud Score", value=f"{fraud_score:.4f}", delta=f"Threshold: {threshold:.4f}")

st.sidebar.markdown("---")
st.sidebar.info("Hybrid GNN+XGBoost Fraud Detector")
