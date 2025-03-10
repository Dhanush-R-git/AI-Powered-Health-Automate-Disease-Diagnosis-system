import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensure model_saved directory exists
os.makedirs("model_saved", exist_ok=True)

def train_and_save_model(dataset_path, model_name):
    """Trains a stacking ensemble model on a dataset and saves it."""
    
    print(f"Training model for: {model_name}...")

    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Ensure last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Data split: {X_train.shape} train samples, {X_test.shape} test samples.")

    # Initialize base models
    rf_model = RandomForestClassifier(random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Fit base models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    print(f"Base models trained for {model_name}.")

    # Create meta features using predictions
    rf_meta_features = rf_model.predict_proba(X_test)
    xgb_meta_features = xgb_model.predict_proba(X_test)

    # Stack meta features
    meta_features = np.column_stack((rf_meta_features, xgb_meta_features))

    # Initialize and train meta model
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, y_test)

    print(f"Meta model trained for {model_name}.")

    # Save trained model
    model_path = f"model_saved/{model_name}.pkl"
    joblib.dump(meta_model, model_path)

    print(f"Model saved: {model_path}")

    # Evaluate model
    stacked_predictions = meta_model.predict(meta_features)
    accuracy = accuracy_score(y_test, stacked_predictions)
    
    print(f"{model_name} Stacking Model Accuracy: {accuracy:.4f}\n")

    return model_path, accuracy

# List of datasets
datasets = {
    "diabetes_model": "dataset/diabetes.csv",
    "heart_model": "dataset/heart.csv",
    "parkinson_model": "dataset/parkinsons.csv"
}

# Train models and track results
results = {}
for model_name, dataset_path in datasets.items():
    model_path, accuracy = train_and_save_model(dataset_path, model_name)
    results[model_name] = {"model_path": model_path, "accuracy": accuracy}

print("Training completed for all datasets.")
print(results)