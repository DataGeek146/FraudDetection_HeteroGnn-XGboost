# predictor.py

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear # GNN components
import xgboost as xgb
import joblib
import os
import traceback

# --- Configuration ---
ARTIFACTS_DIR = "./" # Assume artifacts are in the same directory, or specify a path
GNN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_gnn_for_embeddings_model.pth')
XGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'xgb_model_final.joblib') # Or .json
SCALER_GNN_PATH = os.path.join(ARTIFACTS_DIR, 'scaler_gnn.joblib')
USER_LE_PATH = os.path.join(ARTIFACTS_DIR, 'user_le_gnn.joblib')
MERCHANT_LE_PATH = os.path.join(ARTIFACTS_DIR, 'merchant_le_gnn.joblib')
CAT_LE_DICT_PATH = os.path.join(ARTIFACTS_DIR, 'le_dict_gnn.joblib')
USER_TO_CARDS_MAPPING_PATH = os.path.join(ARTIFACTS_DIR, 'user_to_cards_mapping.joblib')

# GNN Parameters (MUST MATCH THE SAVED GNN MODEL'S TRAINING CONFIGURATION)
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS_EMB = GNN_HIDDEN_CHANNELS
GNN_CLASSIFICATION_OUT_CHANNELS = 2 # Or None if not used in saved model structure
GNN_NUM_LAYERS = 2
GNN_GAT_HEADS = 2
GNN_DROPOUT_RATE_TRAINING = 0.3 # Dropout rate used during GNN training

# Feature names for GNN input (MUST MATCH THE ORDER AND NAMES USED DURING GNN TRAINING)
ALL_TX_NUMERIC_FEATS_GNN = ['Amount', 'Hour', 'DayOfWeek', 'Month_Derived',
                            'TimeSinceLastTxUser_log_seconds', 'UserTxCumulativeCount',
                            'UserAvgTxAmountCumulative', 'UserMerchantTxCount', 'AmountToAvgRatioUser']
TX_CATEGORICAL_FEATS_GNN = ['Use Chip', 'Errors?', 'Merchant City', 'Merchant State', 'MCC']

DEFAULT_SECONDS_SINCE_LAST_TX = 3600 * 24 * 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Global variables for loaded artifacts ---
gnn_model_loaded = None
xgb_model_loaded = None
scaler_gnn_loaded = None
user_le_loaded = None
merchant_le_loaded = None
cat_le_dict_loaded = None
optimal_threshold_loaded = 0.5 # Default, will be loaded or updated
user_to_cards_map_loaded = None

# --- HeteroGNN_GAT_For_Embeddings Class Definition (Copied from your notebook's Cell 8) ---
class HeteroGNN_GAT_For_Embeddings(torch.nn.Module):
    def __init__(self, hidden_channels, emb_out_channels,
                 num_users_for_embedding, num_merchants_for_embedding,
                 num_tx_node_features, dropout_rate, num_gnn_layers=2, gat_heads=4,
                 classification_out_channels=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_gnn_layers = num_gnn_layers
        self.gat_heads = gat_heads
        self.classification_out_channels = classification_out_channels

        # Handle potential 0 for embedding table size gracefully (though should be >0)
        if num_users_for_embedding <= 0:
            print("Warning: num_users_for_embedding is <= 0. Setting to 1 for Embedding layer.")
            num_users_for_embedding = 1
        self.user_emb = torch.nn.Embedding(num_users_for_embedding, hidden_channels)

        if num_merchants_for_embedding <= 0:
            print("Warning: num_merchants_for_embedding is <= 0. Setting to 1 for Embedding layer.")
            num_merchants_for_embedding = 1
        self.merchant_emb = torch.nn.Embedding(num_merchants_for_embedding, hidden_channels)

        if num_tx_node_features > 0:
            self.tx_lin = Linear(num_tx_node_features, hidden_channels)
            self.has_tx_features = True
        else:
            self.has_tx_features = False
            self.tx_base_emb = torch.nn.Parameter(torch.randn(1, hidden_channels))

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_gnn_layers):
            conv_dict_layer = {
                key_type: GATConv(
                    (-1, -1), hidden_channels // self.gat_heads,
                    heads=self.gat_heads, dropout=self.dropout_rate,
                    concat=True, add_self_loops=False
                ) for key_type in [('user', 'performs', 'transaction'),
                                   ('transaction', 'to', 'merchant'),
                                   ('transaction', 'performed_by', 'user'),
                                   ('merchant', 'received_from', 'transaction')]
            }
            self.convs.append(HeteroConv(conv_dict_layer, aggr='sum'))
        
        self.emb_projection_lin = Linear(hidden_channels, emb_out_channels)

        if self.classification_out_channels is not None:
            self.classifier_head = Linear(emb_out_channels, self.classification_out_channels)

    def forward(self, x_dict, edge_index_dict, output_type='embeddings'):
        x_dict_transformed = {}
        # User Embedding
        if 'user' in x_dict and x_dict['user'] is not None and self.user_emb.num_embeddings > 0 and x_dict['user'].numel() > 0:
            user_indices = x_dict['user'].squeeze(-1) if x_dict['user'].ndim > 1 and x_dict['user'].shape[-1] == 1 else x_dict['user']
            user_indices = torch.clamp(user_indices, 0, self.user_emb.num_embeddings - 1) # Ensure in range
            x_dict_transformed['user'] = self.user_emb(user_indices)
        
        # Merchant Embedding
        if 'merchant' in x_dict and x_dict['merchant'] is not None and self.merchant_emb.num_embeddings > 0 and x_dict['merchant'].numel() > 0:
            merchant_indices = x_dict['merchant'].squeeze(-1) if x_dict['merchant'].ndim > 1 and x_dict['merchant'].shape[-1] == 1 else x_dict['merchant']
            merchant_indices = torch.clamp(merchant_indices, 0, self.merchant_emb.num_embeddings - 1) # Ensure in range
            x_dict_transformed['merchant'] = self.merchant_emb(merchant_indices)
        
        # Transaction Features
        if 'transaction' in x_dict and x_dict['transaction'] is not None:
            if self.has_tx_features and x_dict['transaction'].shape[1] > 0 :
                x_dict_transformed['transaction'] = F.relu(self.tx_lin(x_dict['transaction']))
            elif not self.has_tx_features:
                num_current_tx_nodes = x_dict['transaction'].shape[0] # x_dict['transaction'] could be (N,0)
                if num_current_tx_nodes > 0: x_dict_transformed['transaction'] = self.tx_base_emb.repeat(num_current_tx_nodes, 1)
            # If has_tx_features is True but x_dict['transaction'] is (N,0), it's handled by the check above.
            elif x_dict['transaction'].shape[1] == 0 and self.has_tx_features: # No actual features passed, but tx_lin exists
                 num_current_tx_nodes = x_dict['transaction'].shape[0]
                 if num_current_tx_nodes > 0: # Use a zero tensor if tx_lin expects features but none given
                     x_dict_transformed['transaction'] = torch.zeros(num_current_tx_nodes, self.tx_lin.in_features, device=self.tx_lin.weight.device)
                     x_dict_transformed['transaction'] = F.relu(self.tx_lin(x_dict_transformed['transaction']))


        h_dict = x_dict_transformed
        for i, conv_layer in enumerate(self.convs):
            try:
                h_dict = conv_layer(h_dict, edge_index_dict)
            except Exception as e_conv: # Catch potential errors if a node type is missing in h_dict
                print(f"Error in GNN conv layer {i}: {e_conv}")
                print(f"h_dict keys before error: {list(h_dict.keys()) if isinstance(h_dict, dict) else 'h_dict not a dict'}")
                # Provide a default empty tensor for transaction if conv fails mid-way
                num_tx_nodes_output = x_dict.get('transaction', torch.empty(0)).shape[0]
                return {'transaction': torch.empty(num_tx_nodes_output, self.emb_projection_lin.out_features, 
                                                 device=self.emb_projection_lin.weight.device)}
            
            if i < self.num_gnn_layers -1 : 
                h_dict = {key: F.dropout(F.relu(val), p=self.dropout_rate, training=self.training) for key, val in h_dict.items()}
            else: 
                h_dict = {key: F.relu(val) for key, val in h_dict.items()}
        
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
            return {'transaction': torch.empty(num_tx_nodes_output, self.emb_projection_lin.out_features, 
                                             device=self.emb_projection_lin.weight.device)}
# --- End of HeteroGNN_GAT_For_Embeddings Class Definition ---

def load_artifacts():
    global gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, user_le_loaded, \
           merchant_le_loaded, cat_le_dict_loaded, optimal_threshold_loaded, user_to_cards_map_loaded
    
    print("Loading artifacts for predictor...")
    artifacts_ok = True

    try:
        if os.path.exists(USER_TO_CARDS_MAPPING_PATH):
            user_to_cards_map_loaded = joblib.load(USER_TO_CARDS_MAPPING_PATH)
            print(f"   User-Card mapping loaded. Contains {len(user_to_cards_map_loaded)} users.")
        else:
            print(f"   CRITICAL WARNING: User-Card mapping file not found at {USER_TO_CARDS_MAPPING_PATH}.")
            user_to_cards_map_loaded = {} # Essential for validation
            artifacts_ok = False
    except Exception as e: print(f"Error loading User-Card map: {e}"); user_to_cards_map_loaded = {}; artifacts_ok = False

    try:
        if os.path.exists(USER_LE_PATH): user_le_loaded = joblib.load(USER_LE_PATH); print("   User LE loaded.")
        else: print(f"   CRITICAL WARNING: User LE not found at {USER_LE_PATH}"); user_le_loaded = None; artifacts_ok = False
    except Exception as e: print(f"Error loading User LE: {e}"); user_le_loaded = None; artifacts_ok = False

    try:
        if os.path.exists(MERCHANT_LE_PATH): merchant_le_loaded = joblib.load(MERCHANT_LE_PATH); print("   Merchant LE loaded.")
        else: print(f"   CRITICAL WARNING: Merchant LE not found at {MERCHANT_LE_PATH}"); merchant_le_loaded = None; artifacts_ok = False
    except Exception as e: print(f"Error loading Merchant LE: {e}"); merchant_le_loaded = None; artifacts_ok = False
    
    num_users_at_training_time = len(user_le_loaded.classes_) if user_le_loaded and hasattr(user_le_loaded, 'classes_') else 1
    num_merchants_at_training_time = len(merchant_le_loaded.classes_) if merchant_le_loaded and hasattr(merchant_le_loaded, 'classes_') else 1
    num_gnn_input_features_after_le = len(ALL_TX_NUMERIC_FEATS_GNN) + len(TX_CATEGORICAL_FEATS_GNN)

    try:
        gnn_model_loaded = HeteroGNN_GAT_For_Embeddings(
            hidden_channels=GNN_HIDDEN_CHANNELS, emb_out_channels=GNN_OUT_CHANNELS_EMB,
            classification_out_channels=GNN_CLASSIFICATION_OUT_CHANNELS,
            num_users_for_embedding=num_users_at_training_time,
            num_merchants_for_embedding=num_merchants_at_training_time,
            num_tx_node_features=num_gnn_input_features_after_le,
            dropout_rate=GNN_DROPOUT_RATE_TRAINING, # Instantiate with training dropout
            num_gnn_layers=GNN_NUM_LAYERS, gat_heads=GNN_GAT_HEADS
        ).to(device)
        if os.path.exists(GNN_MODEL_PATH):
            gnn_model_loaded.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
            gnn_model_loaded.eval() # Set to eval mode (disables dropout behavior)
            print("   GNN model loaded and set to eval mode.")
        else: print(f"   CRITICAL ERROR: GNN model file not found at {GNN_MODEL_PATH}"); artifacts_ok = False
    except Exception as e: print(f"Error loading GNN model: {e}"); traceback.print_exc(); gnn_model_loaded = None; artifacts_ok = False

    try:
        if os.path.exists(XGB_MODEL_PATH): xgb_model_loaded = joblib.load(XGB_MODEL_PATH); print("   XGBoost model loaded.")
        else: print(f"   CRITICAL ERROR: XGBoost model file not found at {XGB_MODEL_PATH}"); artifacts_ok = False
    except Exception as e: print(f"Error loading XGB model: {e}"); xgb_model_loaded = None; artifacts_ok = False
    
    try:
        if os.path.exists(SCALER_GNN_PATH): scaler_gnn_loaded = joblib.load(SCALER_GNN_PATH); print("   GNN Scaler loaded.")
        else: print(f"   CRITICAL WARNING: GNN Scaler not found at {SCALER_GNN_PATH}"); artifacts_ok = False # Scaler is essential
    except Exception as e: print(f"Error loading Scaler: {e}"); scaler_gnn_loaded = None; artifacts_ok = False

    try:
        if os.path.exists(CAT_LE_DICT_PATH): cat_le_dict_loaded = joblib.load(CAT_LE_DICT_PATH); print("   Categorical LE dict loaded.")
        else: print(f"   CRITICAL WARNING: Categorical LE dict not found at {CAT_LE_DICT_PATH}"); artifacts_ok = False
    except Exception as e: print(f"Error loading Cat LE dict: {e}"); cat_le_dict_loaded = None; artifacts_ok = False
    
    optimal_threshold_loaded = 0.2129 # Example from your successful XGBoost run
    print(f"   Using optimal threshold: {optimal_threshold_loaded}")
    
    if not artifacts_ok:
        print("CRITICAL: One or more essential artifacts failed to load. Predictions may fail or be unreliable.")
    print("Artifact loading attempt complete.")
    return artifacts_ok


def preprocess_transaction_for_gnn(transaction_data_dict_raw):
    df_tx = pd.DataFrame([transaction_data_dict_raw.copy()])
    # Amount
    df_tx['Amount'] = str(df_tx['Amount'].iloc[0]).replace('$', '').replace(',', '')
    df_tx['Amount'] = pd.to_numeric(df_tx['Amount'], errors='coerce').fillna(0.0)

    # Timestamp and derived
    try:
        df_tx['Timestamp'] = pd.to_datetime(
            f"{df_tx['Year'].iloc[0]}-{int(df_tx['Month'].iloc[0]):02d}-{int(df_tx['Day'].iloc[0]):02d} {df_tx['Time'].iloc[0]}",
            errors='coerce'
        )
    except: df_tx['Timestamp'] = pd.NaT
    if pd.notna(df_tx['Timestamp'].iloc[0]):
        ts = df_tx['Timestamp'].iloc[0]
        df_tx['Hour'], df_tx['DayOfWeek'], df_tx['Month_Derived'] = ts.hour, ts.dayofweek, ts.month
    else: df_tx['Hour'], df_tx['DayOfWeek'], df_tx['Month_Derived'] = -1, -1, -1

    # Engineered features (defaults for single inference)
    df_tx['TimeSinceLastTxUser_log_seconds'] = np.log1p(DEFAULT_SECONDS_SINCE_LAST_TX)
    df_tx['UserTxCumulativeCount'] = 1.0
    df_tx['UserAvgTxAmountCumulative'] = df_tx['Amount'].iloc[0]
    df_tx['UserMerchantTxCount'] = 1.0
    avg_amt = df_tx['UserAvgTxAmountCumulative'].iloc[0]
    df_tx['AmountToAvgRatioUser'] = df_tx['Amount'].iloc[0] / (avg_amt + 1e-6) if avg_amt > 0 else 0.0
    df_tx['AmountToAvgRatioUser'] = np.clip(df_tx['AmountToAvgRatioUser'], 0, 100) # Simple clip

    # Label Encoding for categorical TX features
    if cat_le_dict_loaded:
        for col, le in cat_le_dict_loaded.items():
            val_to_transform = str(df_tx[col].iloc[0]) if col in df_tx else "Unknown" # Default if col missing
            try:
                if val_to_transform in le.classes_:
                    df_tx[col] = le.transform([val_to_transform])[0]
                else: # Handle unseen: map to first class or a pre-defined "unknown" encoding
                    df_tx[col] = le.transform([le.classes_[0]])[0] if len(le.classes_) > 0 else 0 # Use first known as fallback
            except Exception as e_le:
                print(f"Warning: LE error for {col} with value '{val_to_transform}'. Using default 0. Error: {e_le}")
                df_tx[col] = 0 # Fallback
    else: # Fallback if le_dict not loaded (should not happen if load_artifacts checks properly)
        for col in TX_CATEGORICAL_FEATS_GNN: df_tx[col] = 0


    # Assemble features in correct order
    numeric_values = []
    for col in ALL_TX_NUMERIC_FEATS_GNN:
        numeric_values.append(df_tx[col].iloc[0] if col in df_tx else 0.0)
    numeric_features_np = np.array([numeric_values])

    categorical_values = []
    for col in TX_CATEGORICAL_FEATS_GNN:
        categorical_values.append(df_tx[col].iloc[0] if col in df_tx else 0)
    categorical_features_np = np.array([categorical_values])

    # Scale numeric features
    scaled_numeric_features_np = numeric_features_np
    if scaler_gnn_loaded and numeric_features_np.shape[1] > 0:
        if numeric_features_np.shape[1] == scaler_gnn_loaded.n_features_in_:
            scaled_numeric_features_np = scaler_gnn_loaded.transform(numeric_features_np)
        else:
            print(f"Warning: Scaler expected {scaler_gnn_loaded.n_features_in_} features, got {numeric_features_np.shape[1]}. Using unscaled numerics for this tx.")
    
    tx_features_for_gnn_input = np.concatenate([scaled_numeric_features_np, categorical_features_np], axis=1)
    
    # User and Merchant IDs
    user_val_raw = str(df_tx['User'].iloc[0]) if 'User' in df_tx else "UNKNOWN_USER_PREPROC"
    merchant_val_raw = str(df_tx['Merchant Name'].iloc[0]) if 'Merchant Name' in df_tx else "UNKNOWN_MERCHANT_PREPROC"
    
    user_idx = 0 # Default if unseen or LE not loaded
    if user_le_loaded:
        try:
            if user_val_raw in user_le_loaded.classes_: user_idx = user_le_loaded.transform([user_val_raw])[0]
            # else: user_idx remains 0 (or handle as error/specific unknown index)
        except Exception as e_ule: print(f"Warning: User LE error for '{user_val_raw}'. Using default. Error: {e_ule}")
            
    merchant_idx = 0 # Default
    if merchant_le_loaded:
        try:
            if merchant_val_raw in merchant_le_loaded.classes_: merchant_idx = merchant_le_loaded.transform([merchant_val_raw])[0]
        except Exception as e_mle: print(f"Warning: Merchant LE error for '{merchant_val_raw}'. Using default. Error: {e_mle}")

    return tx_features_for_gnn_input, user_idx, merchant_idx


def predict_fraud(transaction_data_dict):
    global user_to_cards_map_loaded # Ensure it's accessible
    if not all([gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, 
                user_le_loaded, merchant_le_loaded, cat_le_dict_loaded,
                user_to_cards_map_loaded is not None]): # Check map is loaded
        print("ERROR in predict_fraud: Essential artifacts not loaded.")
        return {"error": "Model artifacts are not fully loaded.", "status_code": 503}, 503

    user_id_input = str(transaction_data_dict.get('User', '')).strip()
    card_id_input = str(transaction_data_dict.get('Card', '')).strip()

    print(f"DEBUG: Validating User '{user_id_input}' with Card '{card_id_input}'")
    if user_id_input in user_to_cards_map_loaded:
        user_known_cards = user_to_cards_map_loaded[user_id_input]
        print(f"DEBUG: Known cards for user '{user_id_input}': {user_known_cards}")
        if card_id_input not in user_known_cards:
            print(f"DEBUG: Card '{card_id_input}' NOT IN known cards.")
            return {
                "transaction_id": transaction_data_dict.get("TransactionID", "N/A"),
                "error": "Card validation failed: Provided card is not associated with the specified user.",
                "user_provided": user_id_input,
                "card_provided": card_id_input,
            }, 400
        else:
            print(f"DEBUG: Card '{card_id_input}' IS IN known cards. Proceeding to fraud prediction.")
    else:
        print(f"DEBUG: User '{user_id_input}' not found in User-Card mapping. Card validation effectively skipped for this check. Proceeding to fraud prediction.")
    
    try:
        tx_features_gnn_input, user_idx, merchant_idx = preprocess_transaction_for_gnn(transaction_data_dict.copy())
    except Exception as e_preprocess:
        print(f"Error during preprocessing for prediction: {e_preprocess}")
        traceback.print_exc()
        return {"error": "Failed to preprocess transaction data.", "details": str(e_preprocess)}, 500
        
    data_gnn = HeteroData()
    data_gnn['transaction'].x = torch.tensor(tx_features_gnn_input, dtype=torch.float).to(device)
    data_gnn['transaction'].num_nodes = 1
    
    data_gnn['user'].x = torch.tensor([[user_idx]], dtype=torch.long).to(device)
    data_gnn['user'].num_nodes = 1
    data_gnn['merchant'].x = torch.tensor([[merchant_idx]], dtype=torch.long).to(device)
    data_gnn['merchant'].num_nodes = 1

    data_gnn['user', 'performs', 'transaction'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    data_gnn['transaction', 'performed_by', 'user'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    data_gnn['transaction', 'to', 'merchant'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    data_gnn['merchant', 'received_from', 'transaction'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    with torch.no_grad():
        gnn_output = gnn_model_loaded(data_gnn.x_dict, data_gnn.edge_index_dict, output_type='embeddings')
        transaction_embedding = gnn_output['transaction'].cpu().numpy()

    tabular_features_for_xgb = tx_features_gnn_input
    if transaction_embedding.shape[0] == 0 or transaction_embedding.shape[1] == 0:
        print("Warning: GNN returned empty or invalid embedding. Using only tabular features for XGBoost.")
        augmented_features = tabular_features_for_xgb
    else:
        augmented_features = np.concatenate([tabular_features_for_xgb, transaction_embedding], axis=1)

    try:
        xgb_pred_proba = xgb_model_loaded.predict_proba(augmented_features)[:, 1]
        xgb_pred_fraud_score = xgb_pred_proba[0]
    except Exception as e_xgb:
        print(f"Error during XGBoost prediction: {e_xgb}")
        traceback.print_exc()
        return {"error": "XGBoost prediction failed.", "details": str(e_xgb)}, 500

    is_fraud_prediction = 1 if xgb_pred_fraud_score >= optimal_threshold_loaded else 0
    
    return {
        "transaction_id": transaction_data_dict.get("TransactionID", "N/A"),
        "fraud_score": float(xgb_pred_fraud_score),
        "is_fraud_prediction": is_fraud_prediction,
        "threshold_used": float(optimal_threshold_loaded),
        "card_validation_status": "Validated" if user_id_input in user_to_cards_map_loaded and card_id_input in user_to_cards_map_loaded.get(user_id_input, set()) \
                                  else ("User_Not_In_Map" if user_id_input not in user_to_cards_map_loaded else "Card_Not_Validated_For_User")
    }

# --- Example Usage (for testing predictor.py independently) ---
if __name__ == '__main__':
    print("Running predictor.py as main script for testing...")
    artifacts_are_ok = load_artifacts()

    if not artifacts_are_ok or not all([gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, 
                user_le_loaded, merchant_le_loaded, cat_le_dict_loaded, 
                user_to_cards_map_loaded is not None]): # Crucially check user_to_cards_map_loaded
        print("CRITICAL: Not all essential artifacts were loaded. Cannot proceed with tests.")
    else:
        # Ensure user_to_cards_map_loaded is not empty for meaningful tests
        if not user_to_cards_map_loaded:
            print("WARNING: User-Card mapping is empty. Card validation tests will not be meaningful.")
            # Create a dummy mapping for testing structure if it's empty
            user_to_cards_map_loaded = {'0': {'card_A_for_user_0', 'card_B_for_user_0'}, '1': {'card_C_for_user_1'}}
            print("INFO: Using a DUMMY User-Card mapping for testing purposes ONLY.")


        # Test Case 1: Known User, KNOWN Card for that user
        known_user_for_test = '0' # Assume '0' is a user ID string in your LE and mapping
        known_card_for_user = 'card_A_for_user_0' # Assume this card belongs to user '0'

        sample_tx_valid = {
            'User': known_user_for_test, 'Card': known_card_for_user, 
            'Year': 2023, 'Month': 10, 'Day': 26, 'Time': '12:30:45',
            'Amount': "$50.00", 'Use Chip': 'Chip Transaction', 
            'Merchant Name': '12345', # Use a value your merchant_le_loaded knows
            'Merchant City': 'NEW YORK', 'Merchant State': 'NY', 'Zip': '10001',
            'MCC': '5411', 'Errors?': 'No Error', 'TransactionID': "test_tx_valid_card"
        }
        # Ensure Merchant Name and MCC are known to their respective LEs
        if merchant_le_loaded and len(merchant_le_loaded.classes_) > 0: sample_tx_valid['Merchant Name'] = merchant_le_loaded.classes_[0]
        if cat_le_dict_loaded and 'MCC' in cat_le_dict_loaded and len(cat_le_dict_loaded['MCC'].classes_) > 0:
             sample_tx_valid['MCC'] = cat_le_dict_loaded['MCC'].classes_[0]

        print("\n--- Test Case 1: Valid User-Card ---")
        prediction_result_valid, status_valid = predict_fraud(sample_tx_valid)
        print(f"Status: {status_valid}, Prediction: {prediction_result_valid}")

        # Test Case 2: Known User, WRONG Card for that user
        if known_user_for_test in user_to_cards_map_loaded: # Ensure user exists in map for this test
            sample_tx_invalid_card = sample_tx_valid.copy()
            sample_tx_invalid_card['Card'] = 'WRONG_CARD_999' # A card NOT in user_to_cards_map_loaded[known_user_for_test]
            sample_tx_invalid_card['TransactionID'] = "test_tx_invalid_card"
            print("\n--- Test Case 2: User with WRONG Card ---")
            prediction_result_invalid_card, status_invalid = predict_fraud(sample_tx_invalid_card)
            print(f"Status: {status_invalid}, Prediction: {prediction_result_invalid_card}")

        # Test Case 3: New User (not in mapping)
        sample_tx_new_user = sample_tx_valid.copy()
        sample_tx_new_user['User'] = 'NEW_USER_XYZ123' # User not in user_to_cards_map_loaded
        sample_tx_new_user['Card'] = 'CARD_FOR_NEW_USER_ABC'
        sample_tx_new_user['TransactionID'] = "test_tx_new_user"
        print("\n--- Test Case 3: NEW User (not in mapping) ---")
        prediction_result_new_user, status_new = predict_fraud(sample_tx_new_user)
        print(f"Status: {status_new}, Prediction: {prediction_result_new_user}")
