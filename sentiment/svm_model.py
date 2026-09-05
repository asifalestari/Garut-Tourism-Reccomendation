import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from config import settings
from feature_extraction.tfidf import fit_tfidf, transform_tfidf
from feature_extraction.vectorizer import save_vectorizer

logger = logging.getLogger("pipeline")

class CalibratedLinearSVMPredictor:
    """
    Enhanced Linear SVM Predictor integrating:
    1. Base LinearSVC with Cost-Sensitive Class Weights
    2. Probability Calibration via CalibratedClassifierCV (Platt scaling / Sigmoid)
    3. Decision Boundary / Threshold Moving for Imbalanced Class Optimization
    """
    def __init__(
        self,
        base_model: LinearSVC,
        calibrated_model: Optional[CalibratedClassifierCV] = None,
        thresholds: Optional[np.ndarray] = None,
        use_threshold_moving: bool = True,
        use_smote: bool = False
    ):
        self.base_model = base_model
        self.calibrated_model = calibrated_model
        self.thresholds = thresholds if thresholds is not None else np.array([1.0/3, 1.0/3, 1.0/3])
        self.use_threshold_moving = use_threshold_moving
        self.use_smote = use_smote
        self.classes_ = getattr(base_model, "classes_", np.array([0, 1, 2]))
        
    @property
    def coef_(self):
        return self.base_model.coef_
        
    @property
    def intercept_(self):
        return self.base_model.intercept_

    def decision_function(self, X):
        return self.base_model.decision_function(X)

    def predict_proba(self, X) -> np.ndarray:
        if self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)
        # Fallback to Softmax over decision_function if no calibrator
        decision_scores = self.decision_function(X)
        exp_scores = np.exp(decision_scores - np.max(decision_scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        if self.use_threshold_moving and self.calibrated_model is not None:
            probs = self.predict_proba(X)
            # Threshold moving: normalize probabilities by class threshold weights
            # Argmax over (P_c / T_c) mathematically yields optimal decision under custom priors
            adjusted_scores = probs / self.thresholds
            return np.argmax(adjusted_scores, axis=1)
        return self.base_model.predict(X)

def train_svm_classifier(df: pd.DataFrame) -> Tuple[CalibratedLinearSVMPredictor, Any, Dict[str, Any]]:
    """
    Handles:
    - Stratified Train/Test Split (80:20, random_state=42)
    - Class distribution analysis
    - TF-IDF fit strictly on Train and transform Train/Test
    - Optional SMOTE Resampling on Train set (leakage-free)
    - Multiclass Linear SVM training (C=0.2, class_weight='balanced')
    - Calibrated Classifier fitting (CalibratedClassifierCV)
    - Threshold-moving predictor packaging
    - Model serialization to models/svm_model.joblib
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
    
    # Optional SMOTE oversampling strictly on training set
    use_smote = getattr(settings, "USE_SMOTE", False)
    X_train_fit = X_train_tfidf
    y_train_fit = y_train
    smote_info = {"applied": False}
    
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            k_neighbors = getattr(settings, "SMOTE_K_NEIGHBORS", 5)
            min_class_samples = min(dist_train.values)
            k = min(k_neighbors, max(1, min_class_samples - 1))
            
            logger.info(f"Applying SMOTE oversampling (k_neighbors={k}, random_state={random_state})...")
            smote = SMOTE(k_neighbors=k, random_state=random_state)
            X_train_fit, y_train_fit = smote.fit_resample(X_train_tfidf, y_train)
            
            dist_resampled = pd.Series(y_train_fit).value_counts().sort_index()
            logger.info(f"SMOTE resampled training distribution: {dict(dist_resampled)}")
            smote_info = {
                "applied": True,
                "k_neighbors": k,
                "resampled_distribution": {label_map[k_idx]: int(v) for k_idx, v in dist_resampled.items()}
            }
        except Exception as e:
            logger.warning(f"SMOTE application skipped due to error: {e}")
            X_train_fit = X_train_tfidf
            y_train_fit = y_train
    
    # Linear SVM Multiclass Classification
    c_param = getattr(settings, "SVM_C", 0.2)
    class_weight = getattr(settings, "SVM_CLASS_WEIGHT", "balanced")
    
    logger.info(f"Training Linear SVM (C={c_param}, class_weight={class_weight})...")
    base_svm = LinearSVC(
        C=c_param,
        class_weight=class_weight,
        random_state=random_state,
        dual=False
    )
    base_svm.fit(X_train_fit, y_train_fit)
    logger.info("Base Linear SVM model training complete.")
    
    # Probability Calibration via 3-Fold CV
    logger.info("Calibrating SVM probabilities using CalibratedClassifierCV (cv=3)...")
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_svm.fit(X_train_fit, y_train_fit)
    logger.info("Probability calibration complete.")
    
    # Threshold configuration
    use_threshold_moving = getattr(settings, "USE_THRESHOLD_MOVING", True)
    thresh_config = getattr(settings, "CLASS_PROB_THRESHOLDS", {0: 0.225, 1: 0.150, 2: 0.625})
    threshold_vector = np.array([
        thresh_config.get(0, 0.225),
        thresh_config.get(1, 0.150),
        thresh_config.get(2, 0.625)
    ], dtype=float)
    
    logger.info(f"Threshold Moving Enabled: {use_threshold_moving} (Thresholds: {threshold_vector.tolist()})")
    
    # Package into predictor object
    predictor = CalibratedLinearSVMPredictor(
        base_model=base_svm,
        calibrated_model=calibrated_svm,
        thresholds=threshold_vector,
        use_threshold_moving=use_threshold_moving,
        use_smote=use_smote
    )
    
    # Save Model objects
    model_path = settings.MODELS_DIR / "svm_model.joblib"
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(predictor, model_path)
    logger.info(f"Calibrated SVM Predictor saved to {model_path}")
    
    save_vectorizer(vectorizer)
    
    # Pack training metadata
    train_meta = {
        "dataset_size": overall_total,
        "train_size": train_total,
        "test_size": test_total,
        "class_weights": class_weight,
        "use_smote": use_smote,
        "smote_info": smote_info,
        "use_threshold_moving": use_threshold_moving,
        "class_thresholds": thresh_config,
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
    
    return predictor, vectorizer, train_meta
