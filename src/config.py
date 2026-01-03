import os

class Config:
    # Paths
    DATA_PATH = "customer_support_tickets.csv"
    ARTIFACTS_DIR = "artifacts"
    
    # Models
    SENTIMENT_MODEL_ID = "roberta-base"
    CATEGORY_MODEL_ID = "distilbert-base-uncased"
    EMBEDDING_MODEL_ID = "distilbert-base-uncased"
    
    # Training Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    SEED = 42
    
    # Mappings
    SENTIMENT_MAP = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    @staticmethod
    def setup():
        """Creates necessary directories for artifacts and logs."""
        os.makedirs(Config.ARTIFACTS_DIR, exist_ok=True)
        # Create sub-directory for TensorBoard logs to avoid path errors during training
        os.makedirs(os.path.join(Config.ARTIFACTS_DIR, "logs"), exist_ok=True)

# Run setup immediately when config is imported to ensure paths exist
Config.setup()