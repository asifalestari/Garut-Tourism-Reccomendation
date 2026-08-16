import logging
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from config import settings

logger = logging.getLogger("pipeline")

def fit_tfidf(train_texts: List[str]) -> TfidfVectorizer:
    """
    Fits a TF-IDF Vectorizer ONLY on the training texts.
    No test texts or overall corpus statistics are learned here, preventing data leakage.
    """
    logger.info("Fitting TF-IDF Vectorizer on training data...")
    max_features = getattr(settings, "TFIDF_MAX_FEATURES", 5000)
    ngram_range = getattr(settings, "TFIDF_NGRAM_RANGE", (1, 2))
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    vectorizer.fit(train_texts)
    
    logger.info(f"TF-IDF Vectorizer fitted with vocabulary size: {len(vectorizer.vocabulary_)}")
    return vectorizer

def transform_tfidf(texts: List[str], vectorizer: TfidfVectorizer):
    """
    Transforms review texts into numerical TF-IDF feature representations
    using an already fitted vectorizer.
    """
    return vectorizer.transform(texts)
