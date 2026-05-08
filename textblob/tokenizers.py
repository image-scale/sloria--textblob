"""Tokenizers for TextBlob."""

from .base import BaseTokenizer
from .decorators import requires_nltk_corpus


class WordTokenizer(BaseTokenizer):
    """Tokenizer that splits text into words using NLTK's TreeBank tokenizer.

    This tokenizer handles:
    - Contractions: "don't" -> ["do", "n't"]
    - Punctuation: separates punctuation from words
    - Special characters and abbreviations
    """

    def __init__(self):
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from nltk.tokenize import TreebankWordTokenizer
            self._tokenizer = TreebankWordTokenizer()
        return self._tokenizer

    def tokenize(self, text, include_punc=True):
        """Tokenize a string into individual words.

        Args:
            text: The text to tokenize
            include_punc: If True (default), include punctuation tokens.
                         If False, filter out punctuation.

        Returns:
            List of word tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        if include_punc:
            return tokens
        else:
            import string
            return [t for t in tokens if t not in string.punctuation]


class SentenceTokenizer(BaseTokenizer):
    """Tokenizer that splits text into sentences using NLTK's Punkt tokenizer."""

    def __init__(self):
        self._tokenizer = None

    @property
    @requires_nltk_corpus('punkt_tab')
    def tokenizer(self):
        if self._tokenizer is None:
            from nltk.tokenize import PunktSentenceTokenizer
            self._tokenizer = PunktSentenceTokenizer()
        return self._tokenizer

    def tokenize(self, text):
        """Tokenize a string into sentences.

        Args:
            text: The text to tokenize

        Returns:
            List of sentence strings.
        """
        return self.tokenizer.tokenize(text)

    def span_tokenize(self, text):
        """Tokenize and return character spans.

        Args:
            text: The text to tokenize

        Returns:
            Generator of (start, end) tuples for each sentence.
        """
        return self.tokenizer.span_tokenize(text)


def word_tokenize(text, include_punc=True):
    """Tokenize text into words.

    This is a convenience function using the default WordTokenizer.

    Args:
        text: The text to tokenize
        include_punc: If True, include punctuation tokens

    Returns:
        List of word tokens.
    """
    tokenizer = WordTokenizer()
    return tokenizer.tokenize(text, include_punc=include_punc)


def sent_tokenize(text):
    """Tokenize text into sentences.

    This is a convenience function using the default SentenceTokenizer.

    Args:
        text: The text to tokenize

    Yields:
        Sentence strings.
    """
    tokenizer = SentenceTokenizer()
    for sentence in tokenizer.tokenize(text):
        yield sentence
