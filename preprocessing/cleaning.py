import re

def clean_text(raw_text: str) -> str:
    """
    Cleans raw text:
    - Removes URLs
    - Removes HTML tags
    - Removes emojis and non-ASCII icons
    - Removes special characters and punctuation
    - Normalizes double spaces and excessive newlines
    - Trims leading/trailing whitespace
    """
    if not isinstance(raw_text, str) or not raw_text:
        return ""

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", raw_text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-ASCII characters (e.g. emojis, maps icons)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Replace special characters and punctuation with space, except alphanumeric and standard whitespace
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize whitespace (replace multiple spaces/newlines/tabs with a single space)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
