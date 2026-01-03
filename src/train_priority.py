import xgboost as xgb
import numpy as np
import joblib
import json
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from .config import Config
from .model_utils import extract_bert_embeddings
import os

def train_priority(train_df, val_df, num_classes):
    print("--- Training Prioritization Model (Hybrid XGBoost) ---")
    
    print("Extracting Embeddings (Tower 1)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    X_train_text = extract_bert_embeddings(train_df['Full_Text'].tolist(), Config.EMBEDDING_MODEL_ID, device=device)
    X_val_text = extract_bert_embeddings(val_df['Full_Text'].tolist(), Config.EMBEDDING_MODEL_ID, device=device)
    
    print("Encoding Metadata (Tower 2)...")
    meta_cols = ['Ticket_Channel', 'Email_Domain', 'Submission_Hour']
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    X_train_meta = ohe.fit_transform(train_df[meta_cols])
    X_val_meta = ohe.transform(val_df[meta_cols])
    
    # Save OHE
    joblib.dump(ohe, f"{Config.ARTIFACTS_DIR}/priority_meta_encoder.joblib")
    
    # Concatenate Features
    X_train = np.hstack((X_train_text, X_train_meta))
    X_val = np.hstack((X_val_text, X_val_meta))
    
    y_train = train_df['Priority_Label']
    y_val = val_df['Priority_Label']
    
    print("Training XGBoost Classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss'
    )
    
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluation
    preds = clf.predict(X_val)
    f1 = f1_score(y_val, preds, average='weighted')
    print(f"Priority Model F1 Score: {f1:.4f}")
    
    # Save Model
    model_path = f"{Config.ARTIFACTS_DIR}/priority_xgboost.json"
    clf.save_model(model_path)
    
    # Save Metrics
    metrics = {"f1_weighted": f1}
    with open(f"{Config.ARTIFACTS_DIR}/priority_metrics.json", "w") as f:
        json.dump(metrics, f)
        
    print(f"Hybrid Model saved to {model_path}")