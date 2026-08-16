import logging
import pandas as pd
from tqdm import tqdm
from preprocessing.cleaning import clean_text
from preprocessing.case_folding import case_folding
from preprocessing.tokenization import tokenize
from preprocessing.stopword_removal import remove_stopwords
from preprocessing.stemming import stem_tokens

logger = logging.getLogger("pipeline")

def preprocess_single_text(text: str) -> str:
    """
    Applies the full preprocessing pipeline to a single text string:
    Cleaning -> Case Folding -> Tokenization -> Stopword Removal -> Stemming -> Join
    """
    if not isinstance(text, str) or not text:
        return ""
    cleaned = clean_text(text)
    folded = case_folding(cleaned)
    tokens = tokenize(folded)
    no_stopwords = remove_stopwords(tokens)
    stemmed = stem_tokens(no_stopwords)
    return " ".join(stemmed)

def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the preprocessing pipeline on the 'review_text' column of the DataFrame.
    Saves results to a new 'cleaned_text' column and filters out empty results.
    """
    logger.info("Starting text preprocessing pipeline...")
    
    if "review_text" not in df.columns:
        raise ValueError("DataFrame does not contain 'review_text' column.")

    df = df.copy()
    
    # Apply row-by-row deterministic preprocessing
    # Since it is row-by-row, it does not cause data leakage
    tqdm.pandas(desc="Preprocessing Reviews")
    df["cleaned_text"] = df["review_text"].progress_apply(preprocess_single_text)
    
    initial_count = len(df)
    # Remove records that ended up with empty text after cleaning/stopword removal
    df = df[df["cleaned_text"].str.strip() != ""]
    final_count = len(df)
    
    logger.info(f"Preprocessing finished. Records: {initial_count} -> {final_count} (filtered {initial_count - final_count} empty reviews).")
    return df
