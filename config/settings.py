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
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"

# Imbalance Handling & Decision Boundary Settings
USE_SMOTE = False  # Set to True if SMOTE oversampling is desired
SMOTE_K_NEIGHBORS = 5
USE_THRESHOLD_MOVING = True
# Optimal decision threshold vector: [Negative (0), Neutral (1), Positive (2)]
# Calibrated probability thresholds to balance Precision, Recall, and Macro F1
CLASS_PROB_THRESHOLDS = {
    0: 0.225,  # Negative threshold (optimizes recall on minority negative reviews)
    1: 0.150,  # Neutral threshold (optimizes sensitivity on minority neutral reviews)
    2: 0.625   # Positive threshold (regularizes over-dominance of positive class)
}

# TF-IDF Settings
TFIDF_MAX_FEATURES = 10000
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
