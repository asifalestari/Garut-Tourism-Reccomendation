import logging
import json
import os
from datetime import datetime
import pandas as pd
from config import settings
from preprocessing.pipeline import run_preprocessing_pipeline
from sentiment.labeling import apply_sentiment_labeling
from sentiment.svm_model import train_svm_classifier
from sentiment.evaluation import evaluate_classifier, run_error_analysis
from sentiment.prediction import predict_dataset_sentiment
from policy.analysis import generate_policy_recommendations

logger = logging.getLogger("pipeline")

def load_and_validate_datasets() -> pd.DataFrame:
    """
    Loads original and repaired reviews, merges them prioritizing repaired reviews,
    applies composite key deduplication, and validates all rows against research constraints.
    """
    reviews_path = settings.RAW_DATA_DIR / "reviews.csv"
    repaired_path = settings.RAW_DATA_DIR / "reviews_repaired.csv"
    
    if not reviews_path.exists():
        raise FileNotFoundError(f"Original reviews file not found at: {reviews_path}")
        
    df_orig = pd.read_csv(reviews_path)
    logger.info(f"Loaded original reviews: {len(df_orig)} records.")
    
    if repaired_path.exists():
        df_rep = pd.read_csv(repaired_path)
        logger.info(f"Loaded repaired reviews: {len(df_rep)} records.")
        
        # Concat prioritizing repaired reviews by putting them first in list
        df_orig["_source"] = "original"
        df_rep["_source"] = "repaired"
        df_merged = pd.concat([df_rep, df_orig], ignore_index=True)
        
        # Deduplication based on composite key: destination_name + author + review_date + review_text
        composite_cols = ["destination_name", "author", "review_date", "review_text"]
        df_temp = df_merged.fillna({col: "" for col in composite_cols})
        
        dup_mask = df_temp.duplicated(subset=composite_cols, keep="first")
        duplicate_count = dup_mask.sum()
        
        df_final_merge = df_merged[~dup_mask].copy()
        df_final_merge = df_final_merge.drop(columns=["_source"], errors="ignore")
        
        logger.info(f"Merge stats: Original={len(df_orig)}, Repaired={len(df_rep)}, Combined={len(df_merged)}")
        logger.info(f"Duplicates removed: {duplicate_count}. Final merged records: {len(df_final_merge)}")
        df_reviews = df_final_merge
    else:
        logger.info("No reviews_repaired.csv found. Using original reviews only.")
        df_reviews = df_orig.copy()
        
    # --- Dataset Validation & Integrity checks ---
    logger.info("Starting dataset validation...")
    
    required_cols = ["destination_name", "author", "rating", "review_date", "review_text"]
    for col in required_cols:
        if col not in df_reviews.columns:
            raise ValueError(f"Dataset is missing required column: '{col}'")
            
    initial_len = len(df_reviews)
    
    # 1. Filter out empty/null review text
    df_reviews = df_reviews.dropna(subset=["review_text"])
    df_reviews = df_reviews[df_reviews["review_text"].astype(str).str.strip() != ""]
    after_text_filter = len(df_reviews)
    
    # 2. Filter out empty/null destination name
    df_reviews = df_reviews.dropna(subset=["destination_name"])
    df_reviews = df_reviews[df_reviews["destination_name"].astype(str).str.strip() != ""]
    after_dest_filter = len(df_reviews)
    
    # 3. Filter rating: numeric and in range 1-5
    df_reviews = df_reviews.dropna(subset=["rating"])
    
    def is_valid_rating(r):
        try:
            val = float(r)
            return 1.0 <= val <= 5.0
        except (ValueError, TypeError):
            return False
            
    valid_rating_mask = df_reviews["rating"].apply(is_valid_rating)
    df_reviews = df_reviews[valid_rating_mask].copy()
    df_reviews["rating"] = df_reviews["rating"].astype(float)
    after_rating_filter = len(df_reviews)
    
    logger.info(f"Validation statistics:")
    logger.info(f"  Initial records            : {initial_len}")
    logger.info(f"  After empty text filter    : {after_text_filter} (removed {initial_len - after_text_filter})")
    logger.info(f"  After empty destination    : {after_dest_filter} (removed {after_text_filter - after_dest_filter})")
    logger.info(f"  After invalid rating filter: {after_rating_filter} (removed {after_dest_filter - after_rating_filter})")
    logger.info(f"  Total invalid records removed: {initial_len - after_rating_filter}")
    
    return df_reviews

def main():
    # Setup clean console and file logging
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOG_FILE_PATH, mode="w", encoding="utf-8")
        ]
    )
    
    logger.info("==================================================")
    logger.info("Starting Research Sentiment Analysis Pipeline Run")
    logger.info("==================================================")
    
    # Step 1: Load, Merge, and Validate Dataset
    df_raw = load_and_validate_datasets()
    
    # Step 2: Run Text Preprocessing
    df_processed = run_preprocessing_pipeline(df_raw)
    
    # Export processed reviews to data/final/processed_reviews.csv
    processed_cols = ["destination_name", "author", "rating", "review_text", "cleaned_text"]
    df_processed[processed_cols].to_csv(settings.FINAL_DATA_DIR / "processed_reviews.csv", index=False)
    logger.info("Exported processed_reviews.csv")
    
    # Step 3: Apply Sentiment Labeling (Individual Review Rating-based)
    df_labeled = apply_sentiment_labeling(df_processed)
    
    # Export labeled reviews to data/final/labeled_reviews.csv
    labeled_cols = ["destination_name", "author", "rating", "review_text", "cleaned_text", "sentiment_label"]
    df_labeled[labeled_cols].to_csv(settings.FINAL_DATA_DIR / "labeled_reviews.csv", index=False)
    logger.info("Exported labeled_reviews.csv")
    
    # Step 4: Model Training (Stratified, TF-IDF fit on Train only, multiclass SVM)
    model, vectorizer, train_meta = train_svm_classifier(df_labeled)
    
    # Step 5: Model Evaluation & Error Analysis (Strictly on Test set)
    eval_metrics = evaluate_classifier(model, vectorizer, train_meta)
    run_error_analysis(model, vectorizer, df_labeled, train_meta)
    
    # Step 6: Full Dataset Prediction Inference (transform only, no fit!)
    df_predictions = predict_dataset_sentiment(df_labeled, model, vectorizer)
    
    # Step 7: Policy Analysis and Aggregation (Destination & Category targets)
    generate_policy_recommendations(df_predictions, eval_metrics=eval_metrics)
    
    # Step 8: Save Experiment Metadata
    save_experiment_metadata(train_meta, eval_metrics)
    
    logger.info("==================================================")
    logger.info("Pipeline Execution Completed Successfully!")
    logger.info("==================================================")

def save_experiment_metadata(train_meta, eval_metrics):
    """
    Exports metadata tracking for reproducible auditability.
    """
    import sklearn
    metadata = {
        "experiment_date": datetime.now().isoformat(),
        "dataset_size": train_meta["dataset_size"],
        "train_size": train_meta["train_size"],
        "test_size": train_meta["test_size"],
        "random_state": 42,
        "class_weights": train_meta["class_weights"],
        "overall_accuracy": eval_metrics["accuracy"],
        "macro_f1": eval_metrics["macro_f1"],
        "weighted_f1": eval_metrics["weighted_f1"],
        "baseline_accuracy": eval_metrics["baseline_accuracy"],
        "baseline_macro_f1": eval_metrics["baseline_macro_f1"],
        "class_distribution": train_meta["class_distribution"],
        "train_class_distribution": train_meta["train_class_distribution"],
        "test_class_distribution": train_meta["test_class_distribution"],
        "tfidf_parameters": {
            "max_features": getattr(settings, "TFIDF_MAX_FEATURES", 5000),
            "ngram_range": getattr(settings, "TFIDF_NGRAM_RANGE", (1, 2))
        },
        "svm_parameters": {
            "C": getattr(settings, "SVM_C", 1.0),
            "kernel": "linear"
        },
        "threshold_configuration": {
            "min_reviews": getattr(settings, "MIN_REVIEWS", 10),
            "negative_threshold": getattr(settings, "NEGATIVE_THRESHOLD", 15.0),
            "positive_threshold": getattr(settings, "POSITIVE_THRESHOLD", 70.0),
            "neutral_threshold": getattr(settings, "NEUTRAL_THRESHOLD", 25.0),
            "average_rating_threshold": getattr(settings, "AVERAGE_RATING_THRESHOLD", 4.0)
        },
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__
        }
    }
    
    meta_path = settings.FINAL_DATA_DIR / "experiment_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Experiment metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
