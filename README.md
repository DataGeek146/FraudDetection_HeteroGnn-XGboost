# Hybrid GNN + XGBoost Fraud Detection System

This project implements a sophisticated fraud detection system leveraging a hybrid approach that combines Graph Neural Networks (GNNs) for rich feature representation and XGBoost for powerful classification. The system is designed to identify potentially fraudulent transactions from a credit card transaction dataset.

**Live Demo (Streamlit App):** [Link to your deployed Streamlit App URL will go here if you deploy it]

## Project Overview

Fraud detection presents unique challenges, primarily due to:
1.  **Extreme Class Imbalance:** Fraudulent transactions are rare compared to legitimate ones.
2.  **Evolving Patterns:** Fraudsters constantly change their tactics.
3.  **Subtle Signals:** Fraudulent activities often try to mimic legitimate behavior.

This project addresses these challenges by:
*   Performing comprehensive **feature engineering** to capture transactional, temporal, and behavioral patterns.
*   Utilizing a **Graph Attention Network (GAT)**, a type of GNN, to learn contextual embeddings for transactions based on their relationships with users and merchants.
*   Employing an **XGBoost classifier** on an augmented feature set (tabular features + GNN embeddings) for robust fraud prediction.
*   Implementing strategies for **handling class imbalance**, including class weighting.
*   Conducting **hyperparameter optimization** for the XGBoost model using Optuna.
*   Performing **threshold tuning** to balance precision and recall based on business-relevant metrics like F1-score and conceptual cost-benefit analysis.
*   Incorporating **model explainability** using SHAP values to understand feature contributions.
*   Adding a **User-Card validation** step for data integrity before prediction.

## Key Features & Novelties

*   **Hybrid GNN-XGBoost Architecture:** Combines the representational power of GNNs for relational data with the classification strength of XGBoost for tabular features. The GNN acts as an advanced feature extractor, providing contextual embeddings that enhance the XGBoost model's performance.
*   **Advanced Feature Engineering:**
    *   **Temporal Dynamics:** Features like `TimeSinceLastTxUser_log_seconds` capture the recency of user activity.
    *   **User Behavioral Profiling (Cumulative):** `UserTxCumulativeCount`, `UserAvgTxAmountCumulative`, and `UserMerchantTxCount` build a baseline of user behavior over time.
    *   **Ratio Features:** `AmountToAvgRatioUser` highlights deviations from a user's typical spending.
*   **User-Card Validation:** An explicit check is implemented to ensure the provided card is associated with the user, adding a layer of input data integrity before fraud prediction.
*   **Systematic Optimization & Evaluation:**
    *   Use of **Optuna** for rigorous XGBoost hyperparameter tuning.
    *   Focus on **AUC-PR** as a primary metric due to class imbalance.
    *   **Threshold Tuning** to find optimal operating points beyond default classification thresholds, including a conceptual cost-benefit analysis framework.
*   **Explainability with SHAP:** SHAP values are used to interpret the final XGBoost model, providing insights into which features (including GNN embedding components) are driving fraud predictions. This is crucial for model trust and identifying potential biases or areas for improvement.

## Project Structure

*   `streamlit_app.py`: The main Streamlit application file for the UI and interaction.
*   `predictor.py`: Contains the core logic for loading models/artifacts, preprocessing input data, generating GNN embeddings, and making XGBoost predictions.
*   `requirements.txt`: Lists all Python dependencies.
*   **Artifacts Directory (e.g., `./` or `artifacts/` - not explicitly created as a folder in this example, assumes files are in root):**
    *   `best_gnn_for_embeddings_model.pth`: Saved GNN model weights.
    *   `xgb_model_final.joblib` (or `.json`): Saved final XGBoost model.
    *   `scaler_gnn.joblib`: Saved StandardScaler for GNN features.
    *   `user_le_gnn.joblib`, `merchant_le_gnn.joblib`: Saved LabelEncoders for user/merchant IDs.
    *   `le_dict_gnn.joblib`: Saved dictionary of LabelEncoders for categorical transaction features.
    *   `user_to_cards_mapping.joblib`: Saved mapping of users to their known cards.
    *   `optimal_threshold_xgb_final.txt`: Saved optimal decision threshold for XGBoost.
*   `GNN_embeddings+XGBoost.ipynb` (or similar): The Jupyter/Colab notebook used for development, training, and experimentation.

## Setup and Running

**1. Clone the Repository:**
   ```bash
   git clone https://github.com/DataGeek146/FraudDetection_HeteroGnn-XGboost.git
   cd FraudDetection_HeteroGnn-XGboost
