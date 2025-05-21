# predictor.py

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear
import xgboost as xgb
import joblib
import os
import traceback

# --- Configuration ---
ARTIFACTS_DIR = "./"
GNN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_gnn_for_embeddings_model.pth')
XGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'xgb_model_final.joblib')
SCALER_GNN_PATH = os.path.join(ARTIFACTS_DIR, 'scaler_gnn.joblib')
USER_LE_PATH = os.path.join(ARTIFACTS_DIR, 'user_le_gnn.joblib')
MERCHANT_LE_PATH = os.path.join(ARTIFACTS_DIR, 'merchant_le_gnn.joblib')
CAT_LE_DICT_PATH = os.path.join(ARTIFACTS_DIR, 'le_dict_gnn.joblib')
USER_TO_CARDS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, 'user_to_cards_mapping.joblib')
OPTIMAL_THRESHOLD_XGB_LOAD_PATH = os.path.join(ARTIFACTS_DIR, 'optimal_threshold_xgb_final.txt') # MODIFIED: Path to load from

# GNN Parameters
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS_EMB = GNN_HIDDEN_CHANNELS
GNN_CLASSIFICATION_OUT_CHANNELS = 2
GNN_NUM_LAYERS = 2
GNN_GAT_HEADS = 2
GNN_DROPOUT_RATE_TRAINING = 0.3

ALL_TX_NUMERIC_FEATS_GNN = ['Amount', 'Hour', 'DayOfWeek', 'Month_Derived',
                            'TimeSinceLastTxUser_log_seconds', 'UserTxCumulativeCount',
                            'UserAvgTxAmountCumulative', 'UserMerchantTxCount', 'AmountToAvgRatioUser']
TX_CATEGORICAL_FEATS_GNN = ['Use Chip', 'Errors?', 'Merchant City', 'Merchant State', 'MCC']
DEFAULT_SECONDS_SINCE_LAST_TX = 3600 * 24 * 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gnn_model_loaded = None
xgb_model_loaded = None
scaler_gnn_loaded = None
user_le_loaded = None
merchant_le_loaded = None
cat_le_dict_loaded = None
optimal_threshold_loaded = 0.5 # Default, will be loaded
user_to_cards_map_loaded = None

# --- HeteroGNN_GAT_For_Embeddings Class Definition (Same as before) ---
class HeteroGNN_GAT_For_Embeddings(torch.nn.Module):
    # ... (Paste your full, corrected HeteroGNN_GAT_For_Embeddings class here) ...
    def __init__(self, hidden_channels, emb_out_channels,
                 num_users_for_embedding, num_merchants_for_embedding,
                 num_tx_node_features, dropout_rate, num_gnn_layers=2, gat_heads=4,
                 classification_out_channels=None):
        super().__init__()
        self.dropout_rate = dropout_rate; self.num_gnn_layers = num_gnn_layers; self.gat_heads = gat_heads
        self.classification_out_channels = classification_out_channels
        if num_users_for_embedding <= 0: num_users_for_embedding = 1
        self.user_emb = torch.nn.Embedding(num_users_for_embedding, hidden_channels)
        if num_merchants_for_embedding <= 0: num_merchants_for_embedding = 1
        self.merchant_emb = torch.nn.Embedding(num_merchants_for_embedding, hidden_channels)
        if num_tx_node_features > 0: self.tx_lin = Linear(num_tx_node_features, hidden_channels); self.has_tx_features = True
        else: self.has_tx_features = False; self.tx_base_emb = torch.nn.Parameter(torch.randn(1, hidden_channels))
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_gnn_layers):
            conv_dict_layer = {
                key_type: GATConv((-1, -1), hidden_channels // self.gat_heads, heads=self.gat_heads, dropout=self.dropout_rate, concat=True, add_self_loops=False)
                for key_type in [('user', 'performs', 'transaction'), ('transaction', 'to', 'merchant'), ('transaction', 'performed_by', 'user'), ('merchant', 'received_from', 'transaction')]
            }
            self.convs.append(HeteroConv(conv_dict_layer, aggr='sum'))
        self.emb_projection_lin = Linear(hidden_channels, emb_out_channels)
        if self.classification_out_channels is not None: self.classifier_head = Linear(emb_out_channels, self.classification_out_channels)

    def forward(self, x_dict, edge_index_dict, output_type='embeddings'):
        x_dict_transformed = {}
        if 'user' in x_dict and x_dict['user'] is not None and self.user_emb.num_embeddings > 0 and x_dict['user'].numel() > 0:
            user_indices = x_dict['user'].squeeze(-1) if x_dict['user'].ndim > 1 and x_dict['user'].shape[-1] == 1 else x_dict['user']
            user_indices = torch.clamp(user_indices, 0, self.user_emb.num_embeddings - 1)
            x_dict_transformed['user'] = self.user_emb(user_indices)
        if 'merchant' in x_dict and x_dict['merchant'] is not None and self.merchant_emb.num_embeddings > 0 and x_dict['merchant'].numel() > 0:
            merchant_indices = x_dict['merchant'].squeeze(-1) if x_dict['merchant'].ndim > 1 and x_dict['merchant'].shape[-1] == 1 else x_dict['merchant']
            merchant_indices = torch.clamp(merchant_indices, 0, self.merchant_emb.num_embeddings - 1)
            x_dict_transformed['merchant'] = self.merchant_emb(merchant_indices)
        if 'transaction' in x_dict and x_dict['transaction'] is not None:
            if self.has_tx_features and x_dict['transaction'].shape[1] > 0 :
                x_dict_transformed['transaction'] = F.relu(self.tx_lin(x_dict['transaction']))
            elif not self.has_tx_features:
                num_current_tx_nodes = x_dict['transaction'].shape[0]
                if num_current_tx_nodes > 0: x_dict_transformed['transaction'] = self.tx_base_emb.repeat(num_current_tx_nodes, 1)
            elif x_dict['transaction'].shape[1] == 0 and self.has_tx_features:
                 num_current_tx_nodes = x_dict['transaction'].shape[0]
                 if num_current_tx_nodes > 0:
                     x_dict_transformed['transaction'] = torch.zeros(num_current_tx_nodes, self.tx_lin.in_features, device=self.tx_lin.weight.device)
                     x_dict_transformed['transaction'] = F.relu(self.tx_lin(x_dict_transformed['transaction']))
        h_dict = x_dict_transformed
        for i, conv_layer in enumerate(self.convs):
            try: h_dict = conv_layer(h_dict, edge_index_dict)
            except Exception as e_conv:
                print(f"Error GNN conv layer {i}: {e_conv}"); print(f"h_dict keys: {list(h_dict.keys()) if isinstance(h_dict, dict) else 'h_dict not dict'}")
                num_tx_nodes_output = x_dict.get('transaction', torch.empty(0)).shape[0]
                return {'transaction': torch.empty(num_tx_nodes_output, self.emb_projection_lin.out_features, device=self.emb_projection_lin.weight.device)}
            if i < self.num_gnn_layers -1 : h_dict = {key: F.dropout(F.relu(val), p=self.dropout_rate, training=self.training) for key, val in h_dict.items()}
            else: h_dict = {key: F.relu(val) for key, val in h_dict.items()}
        if 'transaction' in h_dict:
            transaction_gnn_output = h_dict['transaction']
            transaction_embeddings = self.emb_projection_lin(transaction_gnn_output)
            if output_type == 'embeddings': return {'transaction': transaction_embeddings}
            elif output_type == 'classification' and self.classification_out_channels is not None:
                class_input = F.dropout(transaction_embeddings, p=self.dropout_rate, training=self.training)
                return self.classifier_head(class_input)
            else: return {'transaction': transaction_embeddings}
        else:
            num_tx_nodes_output = 0
            if 'transaction' in x_dict and x_dict['transaction'] is not None: num_tx_nodes_output = x_dict['transaction'].shape[0]
            elif 'transaction' in x_dict_transformed and x_dict_transformed['transaction'] is not None: num_tx_nodes_output = x_dict_transformed['transaction'].shape[0]
            return {'transaction': torch.empty(num_tx_nodes_output, self.emb_projection_lin.out_features, device=self.emb_projection_lin.weight.device)}


def load_artifacts():
    global gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, user_le_loaded, \
           merchant_le_loaded, cat_le_dict_loaded, optimal_threshold_loaded, user_to_cards_map_loaded
    print("Loading artifacts for predictor...")
    artifacts_ok = True
    try:
        if os.path.exists(USER_TO_CARDS_MAPPING_PATH):
            user_to_cards_map_loaded = joblib.load(USER_TO_CARDS_MAPPING_PATH)
            print(f"   User-Card mapping loaded. Contains {len(user_to_cards_map_loaded)} users.")
        else: print(f"CRITICAL WARNING: User-Card map NOT FOUND. Card validation will be skipped."); user_to_cards_map_loaded = {}; # No longer sets artifacts_ok to False, predict_fraud handles it
    except Exception as e: print(f"Error loading User-Card map: {e}"); user_to_cards_map_loaded = {}
    try:
        if os.path.exists(USER_LE_PATH): user_le_loaded = joblib.load(USER_LE_PATH); print("   User LE loaded.")
        else: print(f"CRITICAL WARNING: User LE NOT FOUND."); user_le_loaded = None; artifacts_ok = False
    except Exception as e: print(f"Error loading User LE: {e}"); user_le_loaded = None; artifacts_ok = False
    try:
        if os.path.exists(MERCHANT_LE_PATH): merchant_le_loaded = joblib.load(MERCHANT_LE_PATH); print("   Merchant LE loaded.")
        else: print(f"CRITICAL WARNING: Merchant LE NOT FOUND."); merchant_le_loaded = None; artifacts_ok = False
    except Exception as e: print(f"Error loading Merchant LE: {e}"); merchant_le_loaded = None; artifacts_ok = False
    
    num_users_at_training_time = len(user_le_loaded.classes_) if user_le_loaded and hasattr(user_le_loaded, 'classes_') else 1
    num_merchants_at_training_time = len(merchant_le_loaded.classes_) if merchant_le_loaded and hasattr(merchant_le_loaded, 'classes_') else 1
    num_gnn_input_features_after_le = len(ALL_TX_NUMERIC_FEATS_GNN) + len(TX_CATEGORICAL_FEATS_GNN)

    try:
        gnn_model_loaded = HeteroGNN_GAT_For_Embeddings( GNN_HIDDEN_CHANNELS, GNN_OUT_CHANNELS_EMB, GNN_CLASSIFICATION_OUT_CHANNELS, num_users_at_training_time, num_merchants_at_training_time, num_gnn_input_features_after_le, GNN_DROPOUT_RATE_TRAINING, GNN_NUM_LAYERS, GNN_GAT_HEADS).to(device)
        if os.path.exists(GNN_MODEL_PATH): gnn_model_loaded.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device)); gnn_model_loaded.eval(); print("   GNN model loaded and set to eval mode.")
        else: print(f"   CRITICAL ERROR: GNN model file NOT FOUND."); artifacts_ok = False
    except Exception as e: print(f"Error loading GNN model: {e}"); traceback.print_exc(); gnn_model_loaded = None; artifacts_ok = False
    try:
        if os.path.exists(XGB_MODEL_PATH): xgb_model_loaded = joblib.load(XGB_MODEL_PATH); print("   XGBoost model loaded.")
        else: print(f"   CRITICAL ERROR: XGBoost model file NOT FOUND."); artifacts_ok = False
    except Exception as e: print(f"Error loading XGB model: {e}"); xgb_model_loaded = None; artifacts_ok = False
    try:
        if os.path.exists(SCALER_GNN_PATH): scaler_gnn_loaded = joblib.load(SCALER_GNN_PATH); print("   GNN Scaler loaded.")
        else: print(f"   CRITICAL WARNING: GNN Scaler NOT FOUND."); artifacts_ok = False
    except Exception as e: print(f"Error loading Scaler: {e}"); scaler_gnn_loaded = None; artifacts_ok = False
    try:
        if os.path.exists(CAT_LE_DICT_PATH): cat_le_dict_loaded = joblib.load(CAT_LE_DICT_PATH); print("   Categorical LE dict loaded.")
        else: print(f"   CRITICAL WARNING: Categorical LE dict NOT FOUND."); artifacts_ok = False
    except Exception as e: print(f"Error loading Cat LE dict: {e}"); cat_le_dict_loaded = None; artifacts_ok = False
    
    # MODIFIED: Load optimal threshold from file
    if os.path.exists(OPTIMAL_THRESHOLD_XGB_LOAD_PATH):
        try:
            with open(OPTIMAL_THRESHOLD_XGB_LOAD_PATH, 'r') as f:
                optimal_threshold_loaded = float(f.read().strip())
            print(f"   Loaded optimal XGBoost threshold: {optimal_threshold_loaded:.4f}")
        except Exception as e_thresh:
            print(f"   WARNING: Could not read optimal threshold file. Using default 0.5. Error: {e_thresh}")
            optimal_threshold_loaded = 0.5
            # artifacts_ok = False # Decided not to make this critical for loading, predict_fraud will use default
    else:
        print(f"   WARNING: Optimal threshold file NOT FOUND at {OPTIMAL_THRESHOLD_XGB_LOAD_PATH}. Using default 0.5.")
        optimal_threshold_loaded = 0.5
            
    if not artifacts_ok: print("CRITICAL: One or more essential artifacts failed to load. Predictions may be unreliable or fail.")
    print("Artifact loading attempt complete.")
    return artifacts_ok

# ... (preprocess_transaction_for_gnn - no changes from last version) ...
def preprocess_transaction_for_gnn(transaction_data_dict_raw):
    df_tx = pd.DataFrame([transaction_data_dict_raw.copy()])
    df_tx['Amount'] = str(df_tx['Amount'].iloc[0]).replace('$', '').replace(',', '')
    df_tx['Amount'] = pd.to_numeric(df_tx['Amount'], errors='coerce').fillna(0.0)
    try:
        df_tx['Timestamp'] = pd.to_datetime(
            f"{df_tx['Year'].iloc[0]}-{int(df_tx['Month'].iloc[0]):02d}-{int(df_tx['Day'].iloc[0]):02d} {df_tx['Time'].iloc[0]}",
            errors='coerce')
    except: df_tx['Timestamp'] = pd.NaT
    if pd.notna(df_tx['Timestamp'].iloc[0]): ts = df_tx['Timestamp'].iloc[0]; df_tx['Hour'], df_tx['DayOfWeek'], df_tx['Month_Derived'] = ts.hour, ts.dayofweek, ts.month
    else: df_tx['Hour'], df_tx['DayOfWeek'], df_tx['Month_Derived'] = -1, -1, -1
    df_tx['TimeSinceLastTxUser_log_seconds'] = np.log1p(DEFAULT_SECONDS_SINCE_LAST_TX)
    df_tx['UserTxCumulativeCount'] = 1.0
    df_tx['UserAvgTxAmountCumulative'] = df_tx['Amount'].iloc[0]
    df_tx['UserMerchantTxCount'] = 1.0
    avg_amt = df_tx['UserAvgTxAmountCumulative'].iloc[0]
    df_tx['AmountToAvgRatioUser'] = df_tx['Amount'].iloc[0] / (avg_amt + 1e-6) if avg_amt !=0 else 0.0
    df_tx['AmountToAvgRatioUser'] = np.clip(df_tx['AmountToAvgRatioUser'], 0, 100)
    if cat_le_dict_loaded:
        for col, le in cat_le_dict_loaded.items():
            val_to_transform = str(df_tx[col].iloc[0]) if col in df_tx else "Unknown"
            try:
                if val_to_transform in le.classes_: df_tx[col] = le.transform([val_to_transform])[0]
                else: df_tx[col] = le.transform([le.classes_[0]])[0] if len(le.classes_) > 0 else 0
            except Exception as e_le: print(f"Warn: LE error for {col} val '{val_to_transform}'. Default 0. Err: {e_le}"); df_tx[col] = 0
    else:
        for col in TX_CATEGORICAL_FEATS_GNN: df_tx[col] = 0
    numeric_values = [df_tx[col].iloc[0] if col in df_tx else 0.0 for col in ALL_TX_NUMERIC_FEATS_GNN]
    numeric_features_np = np.array([numeric_values])
    categorical_values = [df_tx[col].iloc[0] if col in df_tx else 0 for col in TX_CATEGORICAL_FEATS_GNN]
    categorical_features_np = np.array([categorical_values])
    scaled_numeric_features_np = numeric_features_np
    if scaler_gnn_loaded and numeric_features_np.shape[1] > 0:
        if numeric_features_np.shape[1] == scaler_gnn_loaded.n_features_in_:
            scaled_numeric_features_np = scaler_gnn_loaded.transform(numeric_features_np)
        else: print(f"Warn: Scaler mismatch. Expected {scaler_gnn_loaded.n_features_in_}, got {numeric_features_np.shape[1]}. Using unscaled.")
    tx_features_for_gnn_input = np.concatenate([scaled_numeric_features_np, categorical_features_np], axis=1)
    user_val_raw = str(df_tx['User'].iloc[0]) if 'User' in df_tx else "UNKNOWN_USER_PREPROC"
    merchant_val_raw = str(df_tx['Merchant Name'].iloc[0]) if 'Merchant Name' in df_tx else "UNKNOWN_MERCHANT_PREPROC"
    user_idx, merchant_idx = 0, 0
    if user_le_loaded:
        try:
            if user_val_raw in user_le_loaded.classes_: user_idx = user_le_loaded.transform([user_val_raw])[0]
        except Exception as e_ule: print(f"Warn: User LE for '{user_val_raw}'. Default. Err: {e_ule}")
    if merchant_le_loaded:
        try:
            if merchant_val_raw in merchant_le_loaded.classes_: merchant_idx = merchant_le_loaded.transform([merchant_val_raw])[0]
        except Exception as e_mle: print(f"Warn: Merch LE for '{merchant_val_raw}'. Default. Err: {e_mle}")
    return tx_features_for_gnn_input, user_idx, merchant_idx

# predict_fraud (no change from previous version where it returns tuple (dict, status_code) for all paths)
def predict_fraud(transaction_data_dict):
    # ... (Keep the version from the previous response that always returns (dict, status_code)) ...
    global user_to_cards_map_loaded 
    if not all([gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, user_le_loaded, merchant_le_loaded, cat_le_dict_loaded, user_to_cards_map_loaded is not None]):
        print("ERROR in predict_fraud: Essential artifacts not loaded.")
        return {"error": "Model artifacts are not fully loaded.", "status_code": 503}, 503
    user_id_input = str(transaction_data_dict.get('User', '')).strip()
    card_id_input = str(transaction_data_dict.get('Card', '')).strip()
    print(f"PREDICTOR DEBUG: Validating User '{user_id_input}' with Card '{card_id_input}'")
    card_val_status_msg = "Card_Not_Validated_Or_Map_Missing_Or_User_New"
    if user_to_cards_map_loaded and user_id_input in user_to_cards_map_loaded:
        user_known_cards = user_to_cards_map_loaded[user_id_input]
        print(f"PREDICTOR DEBUG: Known cards for user '{user_id_input}': {user_known_cards}")
        if card_id_input not in user_known_cards:
            print(f"PREDICTOR DEBUG: Card '{card_id_input}' NOT IN known cards for user '{user_id_input}'.")
            card_val_status_msg = "Card_Not_Associated_With_User"
            return {"transaction_id": transaction_data_dict.get("TransactionID", "N/A"), "error": "Card validation failed: Provided card is not associated with the specified user.", "user_provided": user_id_input, "card_provided": card_id_input,}, 400
        else: print(f"PREDICTOR DEBUG: Card '{card_id_input}' IS IN known cards. Proceeding."); card_val_status_msg = "Validated"
    elif user_to_cards_map_loaded and user_id_input not in user_to_cards_map_loaded: print(f"PREDICTOR DEBUG: User '{user_id_input}' not found in User-Card mapping. Proceeding."); card_val_status_msg = "User_Not_In_Map_Proceeding"
    else: print("PREDICTOR DEBUG: User-Card mapping not loaded/empty. Skipping card validation. Proceeding."); card_val_status_msg = "Map_Missing_Proceeding"
    try: tx_features_gnn_input, user_idx, merchant_idx = preprocess_transaction_for_gnn(transaction_data_dict.copy())
    except Exception as e_preprocess: print(f"Error preprocess: {e_preprocess}"); traceback.print_exc(); return {"error": "Failed to preprocess transaction data.", "details": str(e_preprocess)}, 500
    data_gnn = HeteroData(); data_gnn['transaction'].x = torch.tensor(tx_features_gnn_input, dtype=torch.float).to(device); data_gnn['transaction'].num_nodes = 1
    data_gnn['user'].x = torch.tensor([[user_idx]], dtype=torch.long).to(device); data_gnn['user'].num_nodes = 1
    data_gnn['merchant'].x = torch.tensor([[merchant_idx]], dtype=torch.long).to(device); data_gnn['merchant'].num_nodes = 1
    if gnn_model_loaded:
        if user_idx < gnn_model_loaded.user_emb.num_embeddings : data_gnn['user', 'performs', 'transaction'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device); data_gnn['transaction', 'performed_by', 'user'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
        if merchant_idx < gnn_model_loaded.merchant_emb.num_embeddings: data_gnn['transaction', 'to', 'merchant'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device); data_gnn['merchant', 'received_from', 'transaction'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    with torch.no_grad(): gnn_output = gnn_model_loaded(data_gnn.x_dict, data_gnn.edge_index_dict, output_type='embeddings'); transaction_embedding = gnn_output['transaction'].cpu().numpy()
    tabular_features_for_xgb = tx_features_gnn_input
    if transaction_embedding.shape[0] == 0 or transaction_embedding.shape[1] == 0: augmented_features = tabular_features_for_xgb
    else: augmented_features = np.concatenate([tabular_features_for_xgb, transaction_embedding], axis=1)
    try: xgb_pred_proba = xgb_model_loaded.predict_proba(augmented_features)[:, 1]; xgb_pred_fraud_score = xgb_pred_proba[0]
    except Exception as e_xgb: print(f"Error XGB: {e_xgb}"); traceback.print_exc(); return {"error": "XGBoost prediction failed.", "details": str(e_xgb)}, 500
    is_fraud_prediction = 1 if xgb_pred_fraud_score >= optimal_threshold_loaded else 0
    return {"transaction_id": transaction_data_dict.get("TransactionID", "N/A"), "fraud_score": float(xgb_pred_fraud_score), "is_fraud_prediction": is_fraud_prediction, "threshold_used": float(optimal_threshold_loaded), "card_validation_status": card_val_status_msg}, 200

# --- Example Usage (for testing predictor.py independently) ---
if __name__ == '__main__':
    # ... (Keep your __main__ test block, ensure it calls load_artifacts() first) ...
    print("Running predictor.py as main script for testing...")
    artifacts_are_ok = load_artifacts()
    if not artifacts_are_ok or not all([gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, user_le_loaded, merchant_le_loaded, cat_le_dict_loaded, user_to_cards_map_loaded is not None]):
        print("CRITICAL: Not all essential artifacts were loaded. Exiting test.")
    else:
        if not user_to_cards_map_loaded:
            print("WARNING: User-Card mapping is empty. Card validation tests may not be meaningful for __main__.")
            user_to_cards_map_loaded = {'0': {'card_A_for_user_0', 'card_B_for_user_0'}, '1': {'card_C_for_user_1'}}
            print("INFO: Using a DUMMY User-Card mapping for __main__ testing purposes ONLY.")
        known_user_for_test = '0'; known_card_for_user_A = 'card_A_for_user_0'
        sample_tx_valid = {'User': known_user_for_test, 'Card': known_card_for_user_A, 'Year': 2023, 'Month': 10, 'Day': 26, 'Time': '12:30:45', 'Amount': "$50.00", 'Use Chip': 'Chip Transaction', 'Merchant Name': '12345', 'Merchant City': 'NEW YORK', 'Merchant State': 'NY', 'Zip': '10001', 'MCC': '5411', 'Errors?': 'No Error', 'TransactionID': "test_tx_valid_card"}
        if merchant_le_loaded and len(merchant_le_loaded.classes_) > 0: sample_tx_valid['Merchant Name'] = merchant_le_loaded.classes_[0]
        if cat_le_dict_loaded and 'MCC' in cat_le_dict_loaded and len(cat_le_dict_loaded['MCC'].classes_)>0: sample_tx_valid['MCC'] = cat_le_dict_loaded['MCC'].classes_[0]
        print("\n--- Test Case 1: Valid User-Card ---"); result_dict, status_code = predict_fraud(sample_tx_valid); print(f"Status: {status_code}, Prediction: {result_dict}")
        if user_to_cards_map_loaded and known_user_for_test in user_to_cards_map_loaded:
            sample_tx_invalid_card = sample_tx_valid.copy(); sample_tx_invalid_card['Card'] = 'WRONG_CARD_XYZ'; sample_tx_invalid_card['TransactionID'] = "test_tx_invalid_card"
            print("\n--- Test Case 2: User with WRONG Card ---"); result_dict, status_code = predict_fraud(sample_tx_invalid_card); print(f"Status: {status_code}, Prediction: {result_dict}")
        sample_tx_new_user = sample_tx_valid.copy(); sample_tx_new_user['User'] = 'NEW_USER_TOTALLY_UNKNOWN'; sample_tx_new_user['Card'] = 'CARD_FOR_NEW_USER_456'; sample_tx_new_user['TransactionID'] = "test_tx_new_user"
        print("\n--- Test Case 3: NEW User (not in mapping) ---"); result_dict, status_code = predict_fraud(sample_tx_new_user); print(f"Status: {status_code}, Prediction: {result_dict}")