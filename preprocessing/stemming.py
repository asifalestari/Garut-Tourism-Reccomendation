from typing import List
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize Sastrawi stemmer (reused for all calls)
_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()

def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Stems Indonesian tokens to their base form.
    Joins tokens into a single text block for faster Sastrawi execution.
    """
    if not tokens:
        return []
    sentence = " ".join(tokens)
    stemmed_text = _stemmer.stem(sentence)
    return stemmed_text.split()
