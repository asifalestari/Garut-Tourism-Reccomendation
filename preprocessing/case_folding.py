def case_folding(text: str) -> str:
    """
    Lowercases the input text.
    """
    if not isinstance(text, str) or not text:
        return ""
    return text.lower()
