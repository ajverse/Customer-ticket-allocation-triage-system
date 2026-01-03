import torch
import os
import json
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from .config import Config
from .model_utils import TriageDataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def train_sentiment(train_df, val_df):
    print("--- Training Sentiment Model (RoBERTa) ---")
    
    train_dataset = TriageDataset(
        train_df['Full_Text'].tolist(), 
        train_df['Sentiment_Label'].tolist(), 
        Config.SENTIMENT_MODEL_ID,
        Config.MAX_LEN
    )
    val_dataset = TriageDataset(
        val_df['Full_Text'].tolist(), 
        val_df['Sentiment_Label'].tolist(), 
        Config.SENTIMENT_MODEL_ID,
        Config.MAX_LEN
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.SENTIMENT_MODEL_ID, 
        num_labels=3
    )
    
    training_args = TrainingArguments(
        output_dir=f"{Config.ARTIFACTS_DIR}/sentiment_checkpoints",
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{Config.ARTIFACTS_DIR}/logs/sentiment",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard", # Native PyTorch logging, stable and built-in
        save_total_limit=1       # Keep only the best checkpoint to save space
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # Evaluate & Save Metrics Locally
    eval_results = trainer.evaluate()
    print(f"Sentiment Eval Results: {eval_results}")
    
    save_path = f"{Config.ARTIFACTS_DIR}/sentiment_roberta_model"
    model.save_pretrained(save_path)
    
    # Save metrics to JSON for record keeping
    with open(f"{save_path}/eval_metrics.json", "w") as f:
        json.dump(eval_results, f)
        
    print(f"Sentiment Model saved to {save_path}")