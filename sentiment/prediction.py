import logging
import pandas as pd
from feature_extraction.tfidf import transform_tfidf
from config import settings

logger = logging.getLogger("pipeline")

def predict_dataset_sentiment(df: pd.DataFrame, model, vectorizer) -> pd.DataFrame:
    """
    Runs sentiment predictions (0=Negative, 1=Neutral, 2=Positive) on all valid reviews.
    Uses vectorizer.transform() (DO NOT refit the vectorizer!).
    """
    logger.info("Running full dataset sentiment prediction inference...")
    
    if "cleaned_text" not in df.columns:
        raise ValueError("DataFrame lacks 'cleaned_text' column.")
        
    df = df.copy()
    
    # 1. Transform texts into TF-IDF representation (transform only, no fit!)
    texts = df["cleaned_text"].astype(str).tolist()
    X_tfidf = transform_tfidf(texts, vectorizer)
    
    # 2. Run prediction
    preds = model.predict(X_tfidf)
    
    # 3. Add predictions and text labels to DataFrame
    df["predicted_label"] = preds
    
    sentiment_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    df["predicted_sentiment"] = df["predicted_label"].map(sentiment_map)
    
    # Assert validation
    assert len(df) == len(preds), "Prediction count mismatch with records count."
    
    output_path = settings.FINAL_DATA_DIR / "predicted_reviews.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions complete for {len(df)} reviews. Output saved to {output_path}")
    return df
