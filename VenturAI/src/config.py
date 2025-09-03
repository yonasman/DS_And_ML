import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# API keys
GROK_API_KEY = os.getenv("GROK_API_KEY", "")

# Vector DB settings
VECTOR_DB = os.getenv("VECTOR_DB", "chroma")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", os.path.join(EMBEDDINGS_DIR, "chroma_db"))
