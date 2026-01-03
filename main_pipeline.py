import argparse
from src.config import Config
from src.data_loader import DataLoader
from src.train_sentiment import train_sentiment
from src.train_category import train_categorization
from src.train_priority import train_priority
import torch

def run():
    print("--- Starting Production AI Triage Pipeline ---")
    
    # 1. Data Ingestion & Prep
    loader = DataLoader(Config.DATA_PATH)
    loader.load_data()
    loader.preprocess()
    train_df, val_df = loader.get_splits()
    
    # 2. Train Sentiment (RoBERTa)
    print("\n--- Step 1/3: Sentiment Analysis Model ---")
    train_sentiment(train_df, val_df)
    
    # 3. Train Categorization (DistilBERT + Quantization)
    print("\n--- Step 2/3: Categorization Model (Deployment Optimized) ---")
    num_categories = len(loader.encoders['category'].classes_)
    train_categorization(train_df, val_df, num_categories)
    
    # 4. Train Prioritization (Hybrid XGBoost)
    print("\n--- Step 3/3: Prioritization Model (Hybrid Architecture) ---")
    num_priorities = len(loader.encoders['priority'].classes_)
    train_priority(train_df, val_df, num_priorities)
    
    print("\n--- Pipeline Complete. Artifacts stored in 'artifacts/' ---")

if __name__ == "__main__":
    run()