import torch
import os
import json
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from .config import Config
from .model_utils import TriageDataset, quantize_model

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1': f1}

def train_categorization(train_df, val_df, num_labels):
    print("--- Training Categorization Model (DistilBERT) ---")
    
    train_dataset = TriageDataset(
        train_df['Full_Text'].tolist(), 
        train_df['Category_Label'].tolist(), 
        Config.CATEGORY_MODEL_ID,
        Config.MAX_LEN
    )
    val_dataset = TriageDataset(
        val_df['Full_Text'].tolist(), 
        val_df['Category_Label'].tolist(), 
        Config.CATEGORY_MODEL_ID,
        Config.MAX_LEN
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.CATEGORY_MODEL_ID, 
        num_labels=num_labels
    )
    
    training_args = TrainingArguments(
        output_dir=f"{Config.ARTIFACTS_DIR}/category_checkpoints",
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        eval_strategy="epoch",   # <--- RENAMED FROM evaluation_strategy
        save_strategy="epoch",
        logging_dir=f"{Config.ARTIFACTS_DIR}/logs/category",
        load_best_model_at_end=True,
        report_to="tensorboard",
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    eval_metrics = trainer.evaluate()
    print(f"Categorization Eval Results: {eval_metrics}")
    
    # Save standard model
    full_model_path = f"{Config.ARTIFACTS_DIR}/category_distilbert_full.pt"
    torch.save(model, full_model_path)
    
    # Save metrics
    with open(f"{Config.ARTIFACTS_DIR}/category_metrics.json", "w") as f:
        json.dump(eval_metrics, f)
    
    # QUANTIZATION STEP for Deployment Speed
    quantized_path = f"{Config.ARTIFACTS_DIR}/category_distilbert_quantized.pt"
    quantize_model(full_model_path, quantized_path)
    
    print("Categorization training and quantization complete.")