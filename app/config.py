"""
Configuration for the chatbot backend
"""
import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_CHECKPOINT = BASE_DIR / "checkpoints" / "best_model.pt"
    MAPPINGS_PATH = BASE_DIR / "data" / "mappings.pkl"
    DOMAIN_SCHEMAS_PATH = BASE_DIR / "data" / "domain_schemas.json"
    
    # Model parameters
    VOCAB_SIZE = 30522  # BERT vocab size
    HIDDEN_DIM = 128
    EMBED_DIM = 128
    CONTEXT_WINDOW = 3
    SLIDING_WINDOW = 3
    DROPOUT = 0.3
    NUM_HEADS = 4
    
    # Inference parameters
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    MAX_TOKENS = 128
    
    # Session management
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_HISTORY_TURNS = 10
    
    # Supported domains for rule-based responses
    SUPPORTED_DOMAINS = ["Buses_2", "Movies_1"]
    
    # API settings
    HOST = "0.0.0.0"
    PORT = 8000
    CORS_ORIGINS = ["http://localhost:3000", "http://localhost:3001"]

config = Config()
