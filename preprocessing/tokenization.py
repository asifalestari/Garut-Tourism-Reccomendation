from typing import List

def tokenize(text: str) -> List[str]:
    """
    Splits text into words (tokens) based on whitespace.
    """
    if not isinstance(text, str) or not text:
        return []
    return text.split()
