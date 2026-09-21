"""
Configuration settings for Coffee Disease Diagnosis System
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_DB_PATH = DATA_DIR / "vector_db"
CONFIG_DIR = PROJECT_ROOT / "config"

# PDF Loader settings
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50

# Vector Store settings
VECTOR_DB_TYPE = "faiss"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Retriever settings
RETRIEVER_TOP_K = 5

# Diagnosis settings
MIN_CONFIDENCE_THRESHOLD = 0.8
MAX_QUESTIONS = 5  # Increased from 3 to allow more turns for complex multi-symptom diseases
RELEVANCE_THRESHOLD = 0.3

# Hallucination checker settings
NUM_GENERATIONS_FOR_VERIFICATION = int(os.getenv("NUM_GENERATIONS_FOR_VERIFICATION", "1"))

# LLM provider settings (default to local OpenAI-compatible endpoint, e.g., Ollama)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_local")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "phi3")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

# Anthropic (optional fallback)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "coffee_diagnosis.log"
