"""Utility functions for text processing."""

import re
import string

PUNCTUATION_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")


def strip_punc(s, all=False):
    """Removes punctuation from a string.

    :param s: The string.
    :param all: Remove all punctuation. If False, only removes punctuation from
        the ends of the string.
    """
    if all:
        return PUNCTUATION_REGEX.sub("", s.strip())
    else:
        return s.strip().strip(string.punctuation)


def lowerstrip(s, all=False):
    """Makes text all lowercase and strips punctuation and whitespace.

    :param s: The string.
    :param all: Remove all punctuation. If False, only removes punctuation from
        the ends of the string.
    """
    return strip_punc(s.lower().strip(), all=all)


def is_filelike(obj):
    """Return whether ``obj`` is a file-like object."""
    if not hasattr(obj, "read"):
        return False
    if not callable(obj.read):
        return False
    return True
