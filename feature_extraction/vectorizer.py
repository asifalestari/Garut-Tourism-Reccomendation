import logging
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from config import settings

logger = logging.getLogger("pipeline")

def save_vectorizer(vectorizer: TfidfVectorizer, filepath: Path = None) -> None:
    """
    Saves a fitted TF-IDF Vectorizer object using joblib.
    """
    if filepath is None:
        filepath = settings.MODELS_DIR / "tfidf_vectorizer.joblib"
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, filepath)
    logger.info(f"TF-IDF Vectorizer saved to {filepath}")

def load_vectorizer(filepath: Path = None) -> TfidfVectorizer:
    """
    Loads a saved TF-IDF Vectorizer object using joblib.
    """
    if filepath is None:
        filepath = settings.MODELS_DIR / "tfidf_vectorizer.joblib"
    
    if not filepath.exists():
        raise FileNotFoundError(f"TF-IDF Vectorizer file not found at: {filepath}")
        
    vectorizer = joblib.load(filepath)
    logger.info(f"Loaded TF-IDF Vectorizer from {filepath}")
    return vectorizer
