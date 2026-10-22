import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

PREPROCESSING_VARIANTS = ("clean", "stopwords", "stopwords_stemmed")

_stemmer = PorterStemmer()


def clean_text(text: str) -> str:
    # Remove LaTeX commands such as \alpha before tokenizing the abstract.
    text = re.sub(r"\\[a-zA-Z]+", " ", text.lower())
    # Keep words, spaces, and common punctuation; replace other symbols.
    text = re.sub(r"[^\w\s.,;:!?%()+/-]", " ", text)
    return " ".join(text.split())


def preprocess_text(text: str, variant: str = "clean") -> str:
    if variant not in PREPROCESSING_VARIANTS:
        raise ValueError(f"Unsupported preprocessing variant: {variant}")

    cleaned = clean_text(text)
    if variant == "clean":
        return cleaned

    tokens = re.findall(r"\w+|[^\w\s]", cleaned)
    processed_tokens = []
    for token in tokens:
        if token in ENGLISH_STOP_WORDS:
            continue
        if variant == "stopwords_stemmed" and token.isalpha():
            token = _stemmer.stem(token)
        processed_tokens.append(token)
    return " ".join(processed_tokens)
