# predictor.py

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F # Needed for GNN's forward method
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear # GNN components
import xgboost as xgb
import joblib
import os # For path joining

# --- Configuration ---
# Paths to artifacts - ensure these paths are correct relative to where predictor.py runs
ARTIFACTS_DIR = "./" # Assume artifacts are in the same directory, or specify a path
GNN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_gnn_for_embeddings_model.pth')
XGB_MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'xgb_model_final.joblib') # Or .json if saved differently
SCALER_GNN_PATH = os.path.join(ARTIFACTS_DIR, 'scaler_gnn.joblib')
USER_LE_PATH = os.path.join(ARTIFACTS_DIR, 'user_le_gnn.joblib')
MERCHANT_LE_PATH = os.path.join(ARTIFACTS_DIR, 'merchant_le_gnn.joblib')
CAT_LE_DICT_PATH = os.path.join(ARTIFACTS_DIR, 'le_dict_gnn.joblib')
# OPTIMAL_THRESHOLD_PATH = os.path.join(ARTIFACTS_DIR, 'optimal_threshold.txt') # If saved

# GNN Parameters (MUST MATCH THE SAVED GNN MODEL'S TRAINING CONFIGURATION)
GNN_HIDDEN_CHANNELS = 64
GNN_OUT_CHANNELS_EMB = GNN_HIDDEN_CHANNELS # Embedding output size
GNN_CLASSIFICATION_OUT_CHANNELS = 2 # If GNN had a classifier head, match its output
GNN_NUM_LAYERS = 2
GNN_GAT_HEADS = 2
GNN_DROPOUT_RATE_TRAINING = 0.3 # The dropout rate used during GNN training

# Feature names (MUST MATCH THE ORDER AND NAMES USED DURING GNN TRAINING)
# These are the features that constitute the input to the GNN's self.tx_lin layer (before scaling for numerics)
ALL_TX_NUMERIC_FEATS_GNN = ['Amount', 'Hour', 'DayOfWeek', 'Month_Derived',
                            'TimeSinceLastTxUser_log_seconds', 'UserTxCumulativeCount',
                            'UserAvgTxAmountCumulative', 'UserMerchantTxCount', 'AmountToAvgRatioUser']
TX_CATEGORICAL_FEATS_GNN = ['Use Chip', 'Errors?', 'Merchant City', 'Merchant State', 'MCC']

DEFAULT_SECONDS_SINCE_LAST_TX = 3600 * 24 * 30 # From your notebook

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Global variables for loaded artifacts ---
gnn_model_loaded = None
xgb_model_loaded = None
scaler_gnn_loaded = None
user_le_loaded = None
merchant_le_loaded = None
cat_le_dict_loaded = None
optimal_threshold_loaded = 0.5 # Default, will be loaded or use example

# --- HeteroGNN_GAT_For_Embeddings Class Definition (Copied from notebook) ---
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

        if num_users_for_embedding <= 0: raise ValueError("num_users_for_embedding must be > 0")
        self.user_emb = torch.nn.Embedding(num_users_for_embedding, hidden_channels)

        if num_merchants_for_embedding <= 0: raise ValueError("num_merchants_for_embedding must be > 0")
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
                    (-1, -1),
                    hidden_channels // self.gat_heads,
                    heads=self.gat_heads,
                    dropout=self.dropout_rate, # Dropout on attention coefficients
                    concat=True,
                    add_self_loops=False
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
        if 'user' in x_dict and x_dict['user'] is not None and self.user_emb.num_embeddings > 0:
            user_indices = x_dict['user'].squeeze(-1) if x_dict['user'].ndim > 1 and x_dict['user'].shape[-1] == 1 else x_dict['user']
            if user_indices.numel() > 0 : x_dict_transformed['user'] = self.user_emb(user_indices)
        
        if 'merchant' in x_dict and x_dict['merchant'] is not None and self.merchant_emb.num_embeddings > 0:
            merchant_indices = x_dict['merchant'].squeeze(-1) if x_dict['merchant'].ndim > 1 and x_dict['merchant'].shape[-1] == 1 else x_dict['merchant']
            if merchant_indices.numel() > 0: x_dict_transformed['merchant'] = self.merchant_emb(merchant_indices)
        
        if 'transaction' in x_dict and x_dict['transaction'] is not None:
            if self.has_tx_features and x_dict['transaction'].shape[1] > 0 : # Check if features exist
                x_dict_transformed['transaction'] = F.relu(self.tx_lin(x_dict['transaction']))
            elif not self.has_tx_features: # No features expected, use base_emb
                num_current_tx_nodes = x_dict['transaction'].shape[0] # x_dict['transaction'] could be (N,0)
                if num_current_tx_nodes > 0: x_dict_transformed['transaction'] = self.tx_base_emb.repeat(num_current_tx_nodes, 1)
            # If has_tx_features is True but x_dict['transaction'] is (N,0), tx_lin will fail.
            # This case should be handled by ensuring num_tx_node_features > 0 in __init__ for has_tx_features.

        h_dict = x_dict_transformed
        for i, conv_layer in enumerate(self.convs):
            h_dict = conv_layer(h_dict, edge_index_dict)
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
            # Ensure num_tx_nodes_output is derived safely
            num_tx_nodes_output = 0
            if 'transaction' in x_dict and x_dict['transaction'] is not None:
                 num_tx_nodes_output = x_dict['transaction'].shape[0]
            elif 'transaction' in x_dict_transformed and x_dict_transformed['transaction'] is not None: # If base_emb was used
                 num_tx_nodes_output = x_dict_transformed['transaction'].shape[0]

            return {'transaction': torch.empty(num_tx_nodes_output, self.emb_projection_lin.out_features, 
                                             device=self.emb_projection_lin.weight.device)}
# --- End of HeteroGNN_GAT_For_Embeddings Class Definition ---

def load_artifacts():
    global gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, user_le_loaded, merchant_le_loaded, cat_le_dict_loaded, optimal_threshold_loaded
    
    print("Loading artifacts...")
    # These should be the stats of the data the GNN was *trained* on
    # Load them from a config file or use values from your notebook's Snippet 3 output
    try:
        num_users_at_training_time = len(joblib.load(USER_LE_PATH).classes_) if os.path.exists(USER_LE_PATH) else 10000 # Fallback
        num_merchants_at_training_time = len(joblib.load(MERCHANT_LE_PATH).classes_) if os.path.exists(MERCHANT_LE_PATH) else 50000 # Fallback
    except Exception as e:
        print(f"Warning: Could not load user/merchant LabelEncoder classes count. Using placeholders. Error: {e}")
        num_users_at_training_time = 10000 # Large enough placeholder
        num_merchants_at_training_time = 50000 # Large enough placeholder


    num_gnn_input_features_after_le = len(ALL_TX_NUMERIC_FEATS_GNN) + len(TX_CATEGORICAL_FEATS_GNN)

    # Instantiate with the dropout rate used during training, then call .eval()
    gnn_model_loaded = HeteroGNN_GAT_For_Embeddings(
        hidden_channels=GNN_HIDDEN_CHANNELS,
        emb_out_channels=GNN_OUT_CHANNELS_EMB,
        classification_out_channels=GNN_CLASSIFICATION_OUT_CHANNELS, # Match saved model
        num_users_for_embedding=num_users_at_training_time,
        num_merchants_for_embedding=num_merchants_at_training_time,
        num_tx_node_features=num_gnn_input_features_after_le,
        dropout_rate=GNN_DROPOUT_RATE_TRAINING, # Use training dropout for instantiation
        num_gnn_layers=GNN_NUM_LAYERS,
        gat_heads=GNN_GAT_HEADS
    ).to(device)
    if os.path.exists(GNN_MODEL_PATH):
        gnn_model_loaded.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device))
        gnn_model_loaded.eval() # This sets dropout and batchnorm layers to eval mode
        print("   GNN model loaded and set to eval mode.")
    else: print(f"   ERROR: GNN model file not found at {GNN_MODEL_PATH}")

    if os.path.exists(XGB_MODEL_PATH):
        xgb_model_loaded = joblib.load(XGB_MODEL_PATH)
        print("   XGBoost model loaded.")
    else: print(f"   ERROR: XGBoost model file not found at {XGB_MODEL_PATH}")
    
    if os.path.exists(SCALER_GNN_PATH): scaler_gnn_loaded = joblib.load(SCALER_GNN_PATH); print("   GNN Scaler loaded.")
    else: print(f"   WARNING: GNN Scaler not found at {SCALER_GNN_PATH}")
    
    if os.path.exists(USER_LE_PATH): user_le_loaded = joblib.load(USER_LE_PATH); print("   User LE loaded.")
    else: print(f"   WARNING: User LE not found at {USER_LE_PATH}")

    if os.path.exists(MERCHANT_LE_PATH): merchant_le_loaded = joblib.load(MERCHANT_LE_PATH); print("   Merchant LE loaded.")
    else: print(f"   WARNING: Merchant LE not found at {MERCHANT_LE_PATH}")

    if os.path.exists(CAT_LE_DICT_PATH): cat_le_dict_loaded = joblib.load(CAT_LE_DICT_PATH); print("   Categorical LE dict loaded.")
    else: print(f"   WARNING: Categorical LE dict not found at {CAT_LE_DICT_PATH}")
    
    # Load optimal_threshold if saved, otherwise use a default or example
    # if os.path.exists(OPTIMAL_THRESHOLD_PATH):
    #     with open(OPTIMAL_THRESHOLD_PATH, 'r') as f: optimal_threshold_loaded = float(f.read())
    # else:
    optimal_threshold_loaded = 0.2129 # Example from your notebook
    print(f"   Using optimal threshold: {optimal_threshold_loaded}")
    print("Artifact loading complete.")


def preprocess_transaction_for_gnn(transaction_data_dict_raw):
    # This function prepares features for ONE transaction for GNN input (before GNN embedding layer)
    # It needs to create ALL_TX_NUMERIC_FEATS_GNN and TX_CATEGORICAL_FEATS_GNN
    df_tx = pd.DataFrame([transaction_data_dict_raw.copy()]) # Work on a copy

    # Basic feature cleaning
    if 'Amount' in df_tx.columns:
        if df_tx['Amount'].dtype == 'object' or not pd.api.types.is_numeric_dtype(df_tx['Amount']):
            df_tx['Amount'] = df_tx['Amount'].astype(str).str.replace(r'[$,]', '', regex=True)
        df_tx['Amount'] = pd.to_numeric(df_tx['Amount'], errors='coerce').fillna(0)
    else: df_tx['Amount'] = 0.0


    # Timestamp features
    if 'Year' in df_tx.columns and 'Month' in df_tx.columns and 'Day' in df_tx.columns and 'Time' in df_tx.columns:
        try:
            df_tx['Timestamp'] = pd.to_datetime(
                df_tx['Year'].astype(str) + '-' +
                df_tx['Month'].astype(str).str.zfill(2) + '-' +
                df_tx['Day'].astype(str).str.zfill(2) + ' ' +
                df_tx['Time'].astype(str), errors='coerce'
            )
        except: df_tx['Timestamp'] = pd.NaT
    elif 'Timestamp' in df_tx.columns: # If timestamp is already provided
         df_tx['Timestamp'] = pd.to_datetime(df_tx['Timestamp'], errors='coerce')
    else: df_tx['Timestamp'] = pd.NaT

    if pd.notna(df_tx['Timestamp'].iloc[0]):
        ts = df_tx['Timestamp'].iloc[0]
        df_tx['Hour'] = ts.hour
        df_tx['DayOfWeek'] = ts.dayofweek
        df_tx['Month_Derived'] = ts.month
    else: # Fallback for missing timestamp components
        df_tx['Hour'], df_tx['DayOfWeek'], df_tx['Month_Derived'] = -1, -1, -1

    # Engineered features (defaults for a single transaction, as history is unavailable here)
    df_tx['TimeSinceLastTxUser_log_seconds'] = np.log1p(DEFAULT_SECONDS_SINCE_LAST_TX)
    df_tx['UserTxCumulativeCount'] = 1.0 # First transaction for this user in this context
    df_tx['UserAvgTxAmountCumulative'] = df_tx['Amount'].iloc[0] if 'Amount' in df_tx else 0.0
    df_tx['UserMerchantTxCount'] = 1.0 # First interaction with this merchant for this user
    df_tx['AmountToAvgRatioUser'] = 1.0 if df_tx['UserAvgTxAmountCumulative'].iloc[0] != 0 else 0.0 # Avoid div by zero

    # Apply Label Encoders for categorical transaction features
    if cat_le_dict_loaded:
        for col, le in cat_le_dict_loaded.items():
            if col in df_tx.columns:
                val = str(df_tx[col].iloc[0])
                if val in le.classes_:
                    df_tx[col] = le.transform([val])[0]
                else: # Handle unseen label: map to first class or a dedicated "unknown" index
                    df_tx[col] = le.transform([le.classes_[0]])[0] if len(le.classes_) > 0 else 0
            else: # If column is missing in input, assign a default encoded value (e.g., 0)
                df_tx[col] = 0
    else: # Fallback if le_dict not loaded
        for col in TX_CATEGORICAL_FEATS_GNN: df_tx[col] = 0


    # Assemble numeric and categorical parts
    numeric_part_list = []
    for col in ALL_TX_NUMERIC_FEATS_GNN:
        numeric_part_list.append(df_tx[[col] if col in df_tx else pd.DataFrame({col: [0.0]})].values.flatten())
    numeric_features = np.array(numeric_part_list).T


    categorical_part_list = []
    for col in TX_CATEGORICAL_FEATS_GNN:
         categorical_part_list.append(df_tx[[col] if col in df_tx else pd.DataFrame({col: [0]})].values.flatten())
    categorical_features = np.array(categorical_part_list).T


    # Scale numeric features
    scaled_numeric_features = numeric_features
    if scaler_gnn_loaded and numeric_features.shape[1] > 0 :
        if numeric_features.shape[1] == scaler_gnn_loaded.n_features_in_:
            scaled_numeric_features = scaler_gnn_loaded.transform(numeric_features)
        else:
            print(f"Warning: Scaler expected {scaler_gnn_loaded.n_features_in_} features, got {numeric_features.shape[1]}. Skipping scaling.")
    
    tx_features_for_gnn = np.concatenate([scaled_numeric_features, categorical_features], axis=1)
    
    # User and Merchant IDs
    user_val = str(df_tx['User'].iloc[0]) if 'User' in df_tx else "UNKNOWN_USER"
    merchant_val = str(df_tx['Merchant Name'].iloc[0]) if 'Merchant Name' in df_tx else "UNKNOWN_MERCHANT"
    
    user_idx = -1 # Default for unknown
    if user_le_loaded:
        if user_val in user_le_loaded.classes_: user_idx = user_le_loaded.transform([user_val])[0]
        elif len(user_le_loaded.classes_) > 0 : user_idx = 0 # Map to first known user as fallback
            
    merchant_idx = -1
    if merchant_le_loaded:
        if merchant_val in merchant_le_loaded.classes_: merchant_idx = merchant_le_loaded.transform([merchant_val])[0]
        elif len(merchant_le_loaded.classes_) > 0: merchant_idx = 0 # Map to first known merchant

    return tx_features_for_gnn, user_idx, merchant_idx


def predict_fraud(transaction_data_dict):
    if not all([gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, user_le_loaded, merchant_le_loaded, cat_le_dict_loaded]):
        return {"error": "Model artifacts not loaded. Call load_artifacts() when app starts."}, 500

    # 1. Preprocess for GNN input features
    tx_features_gnn_input, user_idx, merchant_idx = preprocess_transaction_for_gnn(transaction_data_dict.copy())

    # 2. Construct HeteroData for GNN
    data_gnn = HeteroData()
    data_gnn['transaction'].x = torch.tensor(tx_features_gnn_input, dtype=torch.float).to(device)
    data_gnn['transaction'].num_nodes = 1
    
    # User node
    if user_idx != -1 and user_idx < gnn_model_loaded.user_emb.num_embeddings:
        data_gnn['user'].x = torch.tensor([[user_idx]], dtype=torch.long).to(device) # Shape (1,1)
    else: # Fallback for unseen user: use a zero embedding vector directly (or avg, or specific "unknown" emb)
        # For passing to GATConv, if it expects indices, this will be an issue.
        # Safer to ensure user_idx is always valid (e.g., map to an "unknown" ID within embedding range)
        # If GNN structure uses user_emb(x_dict['user']), then user_idx must be valid.
        # Alternative: if user_idx is -1, create a zero tensor for x_dict_transformed['user']
        data_gnn['user'].x = torch.tensor([[0]], dtype=torch.long).to(device) # Default to index 0 if unseen
    data_gnn['user'].num_nodes = 1

    # Merchant node
    if merchant_idx != -1 and merchant_idx < gnn_model_loaded.merchant_emb.num_embeddings:
        data_gnn['merchant'].x = torch.tensor([[merchant_idx]], dtype=torch.long).to(device) # Shape (1,1)
    else:
        data_gnn['merchant'].x = torch.tensor([[0]], dtype=torch.long).to(device) # Default to index 0
    data_gnn['merchant'].num_nodes = 1

    # Edges (transaction is node 0 of its type, user/merchant are node 0 of their type)
    data_gnn['user', 'performs', 'transaction'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    data_gnn['transaction', 'performed_by', 'user'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    data_gnn['transaction', 'to', 'merchant'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)
    data_gnn['merchant', 'received_from', 'transaction'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(device)

    # 3. Generate GNN Embeddings
    with torch.no_grad():
        gnn_output = gnn_model_loaded(data_gnn.x_dict, data_gnn.edge_index_dict, output_type='embeddings')
        transaction_embedding = gnn_output['transaction'].cpu().numpy() # Shape should be (1, emb_dim)

    # 4. Tabular features for XGBoost are the same as GNN input features (already scaled)
    tabular_features_for_xgb = tx_features_gnn_input # Shape (1, num_gnn_input_features)

    # 5. Augment Features
    if transaction_embedding.shape[0] == 0 : # Should not happen if GNN worked
        print("Warning: GNN returned empty embedding. Using only tabular features for XGBoost.")
        augmented_features = tabular_features_for_xgb
    else:
        augmented_features = np.concatenate([tabular_features_for_xgb, transaction_embedding], axis=1)

    # 6. XGBoost Prediction
    xgb_pred_proba = xgb_model_loaded.predict_proba(augmented_features)[:, 1]
    xgb_pred_fraud_score = xgb_pred_proba[0]
    
    # 7. Apply Threshold
    is_fraud_prediction = 1 if xgb_pred_fraud_score >= optimal_threshold_loaded else 0
    
    return {
        "transaction_id": transaction_data_dict.get("TransactionID", "N/A"),
        "fraud_score": float(xgb_pred_fraud_score),
        "is_fraud_prediction": is_fraud_prediction,
        "threshold_used": float(optimal_threshold_loaded)
    }

# --- Example Usage (for testing predictor.py independently) ---
if __name__ == '__main__':
    print("Running predictor.py as main script for testing...")
    # Ensure paths are correct if running this file directly from a different location
    # For this example, assume artifacts are in the same directory as predictor.py
    # ARTIFACTS_DIR = "." # Or specify absolute path if needed

    load_artifacts() # Load all models and preprocessors

    if not all([gnn_model_loaded, xgb_model_loaded, scaler_gnn_loaded, 
                user_le_loaded, merchant_le_loaded, cat_le_dict_loaded]):
        print("CRITICAL: Not all artifacts were loaded. Exiting test.")
    else:
        sample_tx = {
            'User': '0', 'Card': '0', 'Year': 2023, 'Month': 10, 'Day': 26, 'Time': '12:30:45',
            'Amount': "$150.75", 'Use Chip': 'Chip Transaction', 'Merchant Name': '12345',
            'Merchant City': 'NEW YORK', 'Merchant State': 'NY', 'Zip': '10001',
            'MCC': '5411', 'Errors?': 'No Error', 'TransactionID': "test_tx_001"
        }
        # Use known values for User, Merchant, MCC if possible for robust testing of LEs
        if user_le_loaded and len(user_le_loaded.classes_) > 0: sample_tx['User'] = user_le_loaded.classes_[0]
        else: print("Warning: User LE not loaded or empty, using placeholder user for test.")
        if merchant_le_loaded and len(merchant_le_loaded.classes_) > 0: sample_tx['Merchant Name'] = merchant_le_loaded.classes_[0]
        else: print("Warning: Merchant LE not loaded or empty, using placeholder merchant for test.")
        if cat_le_dict_loaded and 'MCC' in cat_le_dict_loaded and len(cat_le_dict_loaded['MCC'].classes_) > 0:
             sample_tx['MCC'] = cat_le_dict_loaded['MCC'].classes_[0]
        else: print("Warning: MCC LE not loaded or empty, using placeholder MCC for test.")


        prediction_result = predict_fraud(sample_tx)
        print("\nPrediction for sample transaction:")
        print(prediction_result)

        sample_tx_new_entities = sample_tx.copy()
        sample_tx_new_entities['User'] = 'NEW_USER_XYZ'
        sample_tx_new_entities['Merchant Name'] = 'NEW_MERCHANT_ABC'
        sample_tx_new_entities['TransactionID'] = "test_tx_002_new_entities"
        prediction_result_new = predict_fraud(sample_tx_new_entities)
        print("\nPrediction for sample transaction with new entities:")
        print(prediction_result_new)