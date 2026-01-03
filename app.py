import torch
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
import numpy as np
import os

from src.config import Config
from src.model_utils import extract_bert_embeddings

class TicketRequest(BaseModel):
    ticket_id: str
    subject: str
    description: str
    email: str
    channel: str

app = FastAPI(title="Advanced AI Triage API", version="2.0")

models = {}
encoders = {}

@app.on_event("startup")
def load_artifacts():
    print("Loading production artifacts...")
    
    # A. Load Encoders
    try:
        global encoders
        encoders = joblib.load(os.path.join(Config.ARTIFACTS_DIR, "encoders.joblib"))
        print("✅ Encoders loaded.")
    except FileNotFoundError:
        print("❌ Encoders not found. Run main_pipeline.py first.")
    
    # B. Load Categorization Model
    try:
        # FIX: Add weights_only=False here as well
        models['category'] = torch.load(
            os.path.join(Config.ARTIFACTS_DIR, "category_distilbert_quantized.pt"),
            map_location='cpu',
            weights_only=False
        )
        models['cat_tokenizer'] = AutoTokenizer.from_pretrained(Config.CATEGORY_MODEL_ID)
        print("✅ Categorization Model loaded.")
    except Exception as e:
        print(f"⚠️ Categorization model failed to load: {e}")

    # C. Load Prioritization Model
    try:
        models['priority'] = xgb.Booster()
        models['priority'].load_model(os.path.join(Config.ARTIFACTS_DIR, "priority_xgboost.json"))
        models['prio_meta_encoder'] = joblib.load(os.path.join(Config.ARTIFACTS_DIR, "priority_meta_encoder.joblib"))
        print("✅ Prioritization Model loaded.")
    except Exception as e:
        print(f"⚠️ Prioritization model failed to load: {e}")


@app.post("/predict")
def predict_ticket(ticket: TicketRequest):
    response = {"ticket_id": ticket.ticket_id}
    full_text = f"{ticket.subject} {ticket.description}"
    
    # CATEGORIZATION
    if 'category' in models:
        tokenizer = models['cat_tokenizer']
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        with torch.no_grad():
            outputs = models['category'](**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            pred_idx = torch.argmax(logits, dim=1).item()
            
        pred_label = encoders['category'].inverse_transform([pred_idx])[0]
        response['category'] = pred_label
    
    # PRIORITIZATION
    if 'priority' in models:
        emb = extract_bert_embeddings([full_text], Config.EMBEDDING_MODEL_ID)
        
        meta_df = pd.DataFrame([{
            'Ticket_Channel': ticket.channel,
            'Email_Domain': ticket.email.split('@')[-1] if '@' in ticket.email else 'unknown',
            'Submission_Hour': pd.Timestamp.now().hour
        }])
        
        meta_enc = models['prio_meta_encoder'].transform(meta_df)
        features = np.hstack((emb, meta_enc))
        dmatrix = xgb.DMatrix(features)
        
        probs = models['priority'].predict(dmatrix)
        pred_idx = np.argmax(probs, axis=1)[0]
        
        pred_prio = encoders['priority'].inverse_transform([pred_idx])[0]
        response['priority'] = pred_prio
        response['priority_confidence'] = float(np.max(probs))

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)