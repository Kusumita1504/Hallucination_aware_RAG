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

# API and service settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_LOG_LEVEL = os.getenv("API_LOG_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
API_MAX_REQUEST_BYTES = int(os.getenv("API_MAX_REQUEST_BYTES", str(64 * 1024)))
API_MAX_SESSIONS = int(os.getenv("API_MAX_SESSIONS", "1000"))
API_SESSION_TTL_SECONDS = int(os.getenv("API_SESSION_TTL_SECONDS", str(60 * 60)))
INCIDENT_DASHBOARD_URL = os.getenv("INCIDENT_DASHBOARD_URL", "http://localhost:3000")
DIAGNOSIS_EVALUATION_FILE = os.getenv("DIAGNOSIS_EVALUATION_FILE")
INCIDENT_DB_PATH = Path(
    os.getenv("INCIDENT_DB_PATH", str(DATA_DIR / "incidents.sqlite3"))
)
INCIDENT_ERROR_RATE_THRESHOLD = float(
    os.getenv("INCIDENT_ERROR_RATE_THRESHOLD", "0.2")
)
INCIDENT_LATENCY_THRESHOLD_SECONDS = float(
    os.getenv("INCIDENT_LATENCY_THRESHOLD_SECONDS", "5")
)
INCIDENT_WINDOW_SECONDS = int(os.getenv("INCIDENT_WINDOW_SECONDS", "300"))
INCIDENT_MIN_REQUESTS = int(os.getenv("INCIDENT_MIN_REQUESTS", "5"))
SUPPORT_LLM_ENABLED = os.getenv("SUPPORT_LLM_ENABLED", "false").lower() == "true"
RUNBOOK_DIR = DATA_DIR / "runbooks"
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")
SLACK_TIMESTAMP_TOLERANCE_SECONDS = int(
    os.getenv("SLACK_TIMESTAMP_TOLERANCE_SECONDS", "300")
)
SLACK_EVENT_DB_PATH = Path(
    os.getenv("SLACK_EVENT_DB_PATH", str(DATA_DIR / "slack_events.sqlite3"))
)

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = PROJECT_ROOT / "coffee_diagnosis.log"
