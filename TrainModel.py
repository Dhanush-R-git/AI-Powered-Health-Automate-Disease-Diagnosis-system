import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure necessary directories exist
os.makedirs("model_saved", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Configure logging
log_file = f"logs/training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def train_and_save_model(dataset_path, model_name, target_column="target", drop_columns=None):
    """Trains a stacking ensemble model on a dataset and saves it, with logging."""
    
    logging.info(f"Training started for: {model_name}")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Drop specified columns if they exist
    if drop_columns:
        df.drop(columns=[col for col in drop_columns if col in df.columns], inplace=True)
    
    # Ensure the target column exists
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in {model_name}. Skipping...")
        return None, None

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info(f"Data split for {model_name}: {X_train.shape} train samples, {X_test.shape} test samples.")

    # Initialize base models
    rf_model = RandomForestClassifier(random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Fit base models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    logging.info(f"Base models trained for {model_name}.")

    # Create meta features using predictions
    rf_meta_features = rf_model.predict_proba(X_test)
    xgb_meta_features = xgb_model.predict_proba(X_test)

    # Stack meta features
    meta_features = np.column_stack((rf_meta_features, xgb_meta_features))

    # Initialize and train meta model
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, y_test)

    logging.info(f"Meta model trained for {model_name}.")

    # Save trained model
    model_path = f"model_saved/{model_name}.sav"
    pickle.dump(meta_model, open(model_path, 'wb'))

    logging.info(f"Model saved: {model_path}")

    # Evaluate model
    stacked_predictions = meta_model.predict(meta_features)
    accuracy = accuracy_score(y_test, stacked_predictions)

    logging.info(f"{model_name} Stacking Model Accuracy: {accuracy:.4f}")
    logging.info(f"Training completed for {model_name}.\n")

    return model_path, accuracy

# List of datasets with target columns and any columns to drop
datasets = {
    "diabetes_model": {"path": "/workspaces/AI-Powered-Health-Automate-Disease-Diagnosis-system/datasets/diabetes.csv", "target": "Outcome"},
    "heart_model": {"path": "/workspaces/AI-Powered-Health-Automate-Disease-Diagnosis-system/datasets/heart.csv", "target": "target"},
    "parkinson_model": {"path": "/workspaces/AI-Powered-Health-Automate-Disease-Diagnosis-system/datasets/parkinsons.csv", "target": "status", "drop": ["name"]}
}

# Train models and log results
results = {}
for model_name, details in datasets.items():
    model_path, accuracy = train_and_save_model(
        dataset_path=details["path"],
        model_name=model_name,
        target_column=details["target"],
        drop_columns=details.get("drop", [])
    )
    if model_path:  # Only store successful results
        results[model_name] = {"model_path": model_path, "accuracy": accuracy}

logging.info("Training completed for all datasets.")
print("Training completed. Check logs for details.")