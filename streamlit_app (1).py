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
        success = predictor.load_artifacts() # This now returns a boolean
        if success:
            st.success("Model artifacts loaded successfully!")
        else:
            st.error("Failed to load some essential model artifacts. Predictions may be unreliable or fail. Check console for details.")
        return success
    except Exception as e:
        st.error(f"Critical error during artifact loading in Streamlit: {e}")
        traceback.print_exc() # Print to console where Streamlit is run
        return False

artifacts_loaded_successfully = load_all_artifacts_streamlit()

# --- App UI ---
st.title("💳 Hybrid GNN + XGBoost Fraud Detection")
st.markdown("Enter transaction details to predict if it's fraudulent.")

if not artifacts_loaded_successfully:
    st.error("Application cannot proceed: Essential model artifacts failed to load. Please check the server console where Streamlit is running.")
    st.stop() # Stop the app if artifacts are not loaded

st.sidebar.header("Transaction Input")
with st.sidebar.form(key='transaction_input_form'):
    default_user_st = "0"
    default_card_st = "card_A_for_user_0"
    default_merchant_st = "12345"
    default_mcc_st = "5411"

    if predictor.user_le_loaded and hasattr(predictor.user_le_loaded, 'classes_') and len(predictor.user_le_loaded.classes_) > 0:
        default_user_st = predictor.user_le_loaded.classes_[0]
        if predictor.user_to_cards_map_loaded and default_user_st in predictor.user_to_cards_map_loaded and predictor.user_to_cards_map_loaded[default_user_st]:
            default_card_st = list(predictor.user_to_cards_map_loaded[default_user_st])[0]
    
    if predictor.merchant_le_loaded and hasattr(predictor.merchant_le_loaded, 'classes_') and len(predictor.merchant_le_loaded.classes_) > 0:
        default_merchant_st = predictor.merchant_le_loaded.classes_[0]
    if predictor.cat_le_dict_loaded and 'MCC' in predictor.cat_le_dict_loaded and hasattr(predictor.cat_le_dict_loaded['MCC'], 'classes_') and len(predictor.cat_le_dict_loaded['MCC'].classes_)>0:
        default_mcc_st = predictor.cat_le_dict_loaded['MCC'].classes_[0]

    user_id = st.text_input("User ID", value=default_user_st)
    card_id = st.text_input("Card ID/Number", value=default_card_st)
    st.markdown("---")
    col1_st, col2_st, col3_st = st.columns(3)
    current_time = pd.Timestamp.now()
    with col1_st: year = st.number_input("Year", 2000, 2050, current_time.year, format="%d")
    with col2_st: month = st.number_input("Month", 1, 12, current_time.month, format="%d")
    with col3_st: day = st.number_input("Day", 1, 31, current_time.day, format="%d")
    transaction_time_str = st.text_input("Time (HH:MM:SS)", value=current_time.strftime("%H:%M:%S"))
    amount_str = st.text_input("Amount (e.g., $123.45 or 123.45)", value="$75.50")
    st.markdown("---")
    use_chip_options = ["Chip Transaction", "Online Transaction", "Swipe Transaction"]
    use_chip = st.selectbox("Transaction Type", use_chip_options, index=0)
    merchant_name_input = st.text_input("Merchant Name/ID", value=default_merchant_st)
    mcc = st.text_input("MCC", value=default_mcc_st)
    merchant_city = st.text_input("Merchant City", value="NEW YORK")
    merchant_state = st.text_input("Merchant State", value="NY")
    zip_code = st.text_input("Zip Code", value="10001")
    errors_options = ["No Error", "Technical Glitch", "Bad PIN", "Insufficient Balance", "Bad CVV", "Bad Card Number", "Bad Expiration"]
    errors = st.selectbox("Errors?", errors_options, index=0)
    submit_button = st.form_submit_button(label='🔍 Predict Fraud')

if submit_button:
    with st.spinner('🧠 Analyzing transaction...'):
        transaction_data_dict = {
            'User': user_id, 'Card': card_id, 'Year': int(year), 'Month': int(month), 'Day': int(day),
            'Time': transaction_time_str, 'Amount': amount_str, 'Use Chip': use_chip,
            'Merchant Name': merchant_name_input, 'Merchant City': merchant_city,
            'Merchant State': merchant_state, 'Zip': zip_code, 'MCC': mcc, 'Errors?': errors,
            'TransactionID': f"st_app_tx_{int(time.time())}"
        }
        st.markdown("---")
        st.markdown("**Input Sent to Predictor:**")
        st.json({k: v for k,v in transaction_data_dict.items() if k != 'TransactionID'})

        response_data, status_code = predictor.predict_fraud(transaction_data_dict)
        
        st.subheader("Prediction Result")
        if "error" in response_data:
            st.error(f"🚫 Error (Status Code {status_code}): {response_data['error']}")
            if "details" in response_data: st.caption(f"Details: {response_data['details']}")
            if "user_provided" in response_data:
                st.warning(f"Input User: {response_data.get('user_provided', 'N/A')}, Input Card: {response_data.get('card_provided', 'N/A')}")
        else:
            fraud_score = response_data.get("fraud_score", 0.0)
            is_fraud = response_data.get("is_fraud_prediction", 0)
            threshold = response_data.get("threshold_used", 0.5)
            card_status = response_data.get("card_validation_status", "Status_Unknown")

            st.info(f"Card Validation Status: {card_status}")
            if is_fraud == 1: st.error(f"🚨 POTENTIALLY FRAUDULENT (Score: {fraud_score:.4f})")
            else: st.success(f"✅ LIKELY NOT FRAUDULENT (Score: {fraud_score:.4f})")
            st.metric(label="Fraud Score", value=f"{fraud_score:.4f}", delta=f"Threshold: {threshold:.4f}")

st.sidebar.markdown("---")
st.sidebar.info("Hybrid GNN+XGBoost Fraud Detector")