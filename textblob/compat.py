"""Python compatibility utilities for TextBlob."""

import sys

PY3 = sys.version_info[0] >= 3
PY310 = sys.version_info >= (3, 10)

string_types = (str,)
text_type = str
binary_type = bytes

def implements_to_string(cls):
    """Class decorator for implementing __str__ in Python 3."""
    return cls
