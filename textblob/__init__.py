"""TextBlob - Simplified Text Processing.

A Python library for processing textual data, providing a simple API
for common natural language processing (NLP) tasks.

Example:
    >>> from textblob import TextBlob
    >>> blob = TextBlob("TextBlob is amazingly simple to use!")
    >>> blob.words
    WordList(['TextBlob', 'is', 'amazingly', 'simple', 'to', 'use'])
    >>> blob.sentiment
    Sentiment(polarity=0.6, subjectivity=0.8)
    >>> blob.noun_phrases
    WordList(['textblob'])
"""

__version__ = '0.1.0'
__author__ = 'TextBlob Clone'

from .blob import TextBlob, Word, WordList, Sentence, Blobber
from .exceptions import (
    TextBlobError,
    MissingCorpusError,
    FormatError,
    TranslatorError,
    NotTranslated,
)

__all__ = [
    'TextBlob',
    'Word',
    'WordList',
    'Sentence',
    'Blobber',
    'TextBlobError',
    'MissingCorpusError',
    'FormatError',
    'TranslatorError',
    'NotTranslated',
]
