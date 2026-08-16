from typing import List
import nltk
from nltk.corpus import stopwords
from config.constants import CUSTOM_INDONESIAN_STOPWORDS

# Load NLTK stopwords with auto-download fallback
try:
    nltk_stopwords = set(stopwords.words("indonesian"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk_stopwords = set(stopwords.words("indonesian"))

# Combine standard and custom stopwords
ALL_STOPWORDS = nltk_stopwords.union(CUSTOM_INDONESIAN_STOPWORDS)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filters out Indonesian standard and custom stopwords from a token list.
    """
    if not tokens:
        return []
    return [token for token in tokens if token not in ALL_STOPWORDS]
