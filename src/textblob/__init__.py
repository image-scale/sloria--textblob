"""TextBlob - Simplified Text Processing

A Python library for processing textual data. It provides a simple API for
diving into common natural language processing (NLP) tasks.
"""

from textblob.blob import Sentence, TextBlob

# Import submodules for easier access
from textblob import taggers

__all__ = [
    "TextBlob",
    "Sentence",
    "taggers",
]
