"""TextBlob - Simplified Text Processing

A Python library for processing textual data. It provides a simple API for
diving into common natural language processing (NLP) tasks.
"""

from textblob.blob import Sentence, TextBlob, Word
from textblob.inflect import pluralize, singularize

# Import submodules for easier access
from textblob import sentiments, taggers

__all__ = [
    "TextBlob",
    "Sentence",
    "Word",
    "sentiments",
    "taggers",
    "pluralize",
    "singularize",
]
