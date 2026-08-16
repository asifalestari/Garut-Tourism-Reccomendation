import logging
import pandas as pd

logger = logging.getLogger("pipeline")

def get_label_from_rating(rating) -> int:
    """
    Pemetaan rating ulasan individu ke label sentimen (rating-based sentiment labeling):
    - 1.0 s/d 2.0 (atau <= 2.5) -> 0 (Negative)
    - 3.0 (atau 2.5 < rating <= 3.5) -> 1 (Neutral)
    - 4.0 s/d 5.0 (atau > 3.5) -> 2 (Positive)
    Returns -1 for invalid or missing ratings.
    """
    try:
        r = float(rating)
        # Check out-of-bounds
        if r < 1.0 or r > 5.0:
            return -1
    except (ValueError, TypeError):
        return -1
        
    if r <= 2.5:
        return 0
    elif r <= 3.5:
        return 1
    else:
        return 2

def apply_sentiment_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rating-based sentiment labeling to the dataset.
    Ensures all records are verified and drops invalid ratings.
    """
    logger.info("Applying individual review rating-based sentiment labeling...")
    
    if "rating" not in df.columns:
        raise ValueError("DataFrame does not contain required 'rating' column.")
        
    df = df.copy()
    df["sentiment_label"] = df["rating"].apply(get_label_from_rating)
    
    initial_len = len(df)
    # Filter out invalid ratings
    df = df[df["sentiment_label"] != -1]
    final_len = len(df)
    
    # Assert that mapping is explicit and all labels are valid
    unique_labels = set(df["sentiment_label"].unique())
    assert unique_labels.issubset({0, 1, 2}), f"Validation failed: found out-of-scope label codes {unique_labels}"
    
    logger.info(f"Sentiment labeling completed. Labeled records: {final_len} (filtered {initial_len - final_len} invalid ratings).")
    return df
