import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from config import settings
from feature_extraction.tfidf import fit_tfidf, transform_tfidf
from feature_extraction.vectorizer import save_vectorizer

logger = logging.getLogger("pipeline")

def train_svm_classifier(df: pd.DataFrame) -> Tuple[LinearSVC, Any, Dict[str, Any]]:
    """
    Handles:
    - Stratified Train/Test Split (80:20, random_state=42)
    - Class distribution analysis
    - TF-IDF fit strictly on Train and transform Train/Test
    - Multiclass Linear SVM training (C=1.0, class_weight='balanced')
    - Model serialization to models/ folder
    """
    logger.info("Initializing train/test split and modeling process...")
    
    if "cleaned_text" not in df.columns or "sentiment_label" not in df.columns:
        raise ValueError("DataFrame lacks required 'cleaned_text' or 'sentiment_label' columns.")
        
    X = df["cleaned_text"].astype(str).tolist()
    y = df["sentiment_label"].astype(int).tolist()
    
    # Check class counts for safety
    class_counts = pd.Series(y).value_counts()
    for label, count in class_counts.items():
        if count < 2:
            raise ValueError(f"Class {label} has insufficient samples ({count}) for stratified split.")
            
    # Stratified split to preserve class proportions
    test_size = getattr(settings, "TEST_SIZE", 0.2)
    random_state = getattr(settings, "RANDOM_STATE", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Class distribution analysis
    overall_total = len(y)
    train_total = len(y_train)
    test_total = len(y_test)
    
    dist_overall = pd.Series(y).value_counts().sort_index()
    dist_train = pd.Series(y_train).value_counts().sort_index()
    dist_test = pd.Series(y_test).value_counts().sort_index()
    
    logger.info("=== Class Sentiment Distribution ===")
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for label in [0, 1, 2]:
        lbl_name = label_map[label]
        count_o = dist_overall.get(label, 0)
        count_tr = dist_train.get(label, 0)
        count_ts = dist_test.get(label, 0)
        
        logger.info(f"  {lbl_name:<10}: Overall={count_o:>5} ({count_o/overall_total*100:.2f}%), "
                    f"Train={count_tr:>5} ({count_tr/train_total*100:.2f}%), "
                    f"Test={count_ts:>5} ({count_ts/test_total*100:.2f}%)")
                    
    # TF-IDF Feature Extraction with Leakage Prevention
    vectorizer = fit_tfidf(X_train)
    X_train_tfidf = transform_tfidf(X_train, vectorizer)
    X_test_tfidf = transform_tfidf(X_test, vectorizer)
    
    # Linear SVM Multiclass Classification
    c_param = getattr(settings, "SVM_C", 1.0)
    class_weight = getattr(settings, "SVM_CLASS_WEIGHT", "balanced")
    
    logger.info(f"Training Linear SVM (C={c_param}, class_weight={class_weight})...")
    model = LinearSVC(
        C=c_param,
        class_weight=class_weight,
        random_state=random_state,
        dual=False  # Recommended when n_samples > n_features for faster convergence
    )
    
    model.fit(X_train_tfidf, y_train)
    logger.info("Linear SVM model training complete.")
    
    # Save Model objects
    model_path = settings.MODELS_DIR / "svm_model.joblib"
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"SVM Model saved to {model_path}")
    
    save_vectorizer(vectorizer)
    
    # Pack training metadata
    train_meta = {
        "dataset_size": overall_total,
        "train_size": train_total,
        "test_size": test_total,
        "class_weights": class_weight,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "class_distribution": {
            "Negative": int(dist_overall.get(0, 0)),
            "Neutral": int(dist_overall.get(1, 0)),
            "Positive": int(dist_overall.get(2, 0))
        },
        "train_class_distribution": {
            "Negative": int(dist_train.get(0, 0)),
            "Neutral": int(dist_train.get(1, 0)),
            "Positive": int(dist_train.get(2, 0))
        },
        "test_class_distribution": {
            "Negative": int(dist_test.get(0, 0)),
            "Neutral": int(dist_test.get(1, 0)),
            "Positive": int(dist_test.get(2, 0))
        }
    }
    
    return model, vectorizer, train_meta
