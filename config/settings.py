import os
from pathlib import Path

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

# Model Storage Directory
MODELS_DIR = BASE_DIR / "models"

# Logs Directory
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "pipeline.log"

# Scraper Settings
HEADLESS = False
LOCALE = "id-ID"
VIEWPORT = {"width": 1280, "height": 720}
DEFAULT_TIMEOUT = 15000  # in milliseconds
BASE_MAPS_URL = "https://www.google.com/maps"
SCROLL_PAUSE_TIME = 2.0  # in seconds
MAX_REVIEWS_PER_DESTINATION = 500
MAX_REVIEW_AGE_MONTHS = 12  # Customizable threshold (e.g. 6 or 12 months)
MAX_REVIEW_MONTHS = 12      # Alias for age limit in months

# Sentiment Analysis Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
SVM_C = 0.2
SVM_KERNEL = "linear"
SVM_CLASS_WEIGHT = None  # Ubah ke None sesuai arahan dosen (tanpa class weight)

# Imbalance Handling & Decision Boundary Settings
USE_SMOTE = False
SMOTE_K_NEIGHBORS = 5
USE_THRESHOLD_MOVING = False  # Threshold moving aktif sebagai metode penyeimbang utama

CLASS_PROB_THRESHOLDS = {
    0: 0.225,  # Threshold Negatif
    1: 0.150,  # Threshold Netral
    2: 0.625   # Threshold Positif
}

# TF-IDF Settings
TFIDF_MAX_FEATURES = 10000  # atau 5000 (pastikan sama dengan di notebook)
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_SUBLINEAR_TF = False

# Policy Classification Operational Thresholds
MIN_REVIEWS = 10
NEGATIVE_THRESHOLD = 15.0
POSITIVE_THRESHOLD = 70.0
NEUTRAL_THRESHOLD = 25.0
AVERAGE_RATING_THRESHOLD = 4.0

# Recommendation Settings
TOP_N = 5

# Create necessary directories if they don't exist
for path in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, FINAL_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
