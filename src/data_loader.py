import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from .config import Config

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.encoders = {}

    def load_data(self):
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath)
        return self.df

    def preprocess(self):
        # 1. Text Construction
        self.df['Full_Text'] = (self.df['Ticket_Subject'] + " " + self.df['Ticket_Description']).fillna("")
        
        # 2. Sentiment Mapping (Satisfaction Score -> Label)
        def map_sentiment(score):
            if score <= 2: return 0 # Negative
            elif score == 3: return 1 # Neutral
            else: return 2 # Positive
        self.df['Sentiment_Label'] = self.df['Satisfaction_Score'].apply(map_sentiment)
        
        # 3. Metadata Engineering
        self.df['Email_Domain'] = self.df['Customer_Email'].apply(lambda x: str(x).split('@')[-1] if '@' in str(x) else 'unknown')
        self.df['Submission_Date'] = pd.to_datetime(self.df['Submission_Date'])
        self.df['Submission_Hour'] = self.df['Submission_Date'].dt.hour
        
        # 4. Label Encoding for Categorization
        le_cat = LabelEncoder()
        self.df['Category_Label'] = le_cat.fit_transform(self.df['Issue_Category'])
        self.encoders['category'] = le_cat
        
        # 5. Label Encoding for Priority
        le_prio = LabelEncoder()
        self.df['Priority_Label'] = le_prio.fit_transform(self.df['Priority_Level'])
        self.encoders['priority'] = le_prio
        
        # Save encoders
        joblib.dump(self.encoders, os.path.join(Config.ARTIFACTS_DIR, "encoders.joblib"))
        
        return self.df

    def get_splits(self):
        # Stratified split based on Category (primary target)
        train_df, val_df = train_test_split(
            self.df, 
            test_size=0.2, 
            random_state=Config.SEED, 
            stratify=self.df['Issue_Category']
        )
        return train_df, val_df