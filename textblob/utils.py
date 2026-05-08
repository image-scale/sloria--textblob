"""Utility functions for TextBlob."""

import string


def strip_punc(s, all=False):
    """Remove punctuation from a string.

    Args:
        s: The string to process
        all: If True, remove all punctuation. If False (default),
             only remove punctuation from the ends.

    Returns:
        String with punctuation removed.
    """
    if all:
        return ''.join(c for c in s if c not in string.punctuation)
    else:
        return s.strip(string.punctuation)


def lowerstrip(s, all=False):
    """Convert to lowercase and strip punctuation.

    Args:
        s: The string to process
        all: If True, remove all punctuation. If False (default),
             only remove punctuation from the ends.

    Returns:
        Lowercase string with punctuation stripped.
    """
    return strip_punc(s.lower().strip(), all=all)


def tree2str(tree, concat=' '):
    """Convert an NLTK tree to a string.

    Args:
        tree: An NLTK Tree object
        concat: String to join the leaves with

    Returns:
        String representation of the tree's leaves.
    """
    try:
        return concat.join(tree.leaves())
    except AttributeError:
        return str(tree)


def filter_insignificant(chunk_list, tag_suffixes=('DT', 'CC', 'PRP$', 'PRP')):
    """Filter out insignificant (function) words from a list of tagged chunks.

    Args:
        chunk_list: List of (word, tag) tuples
        tag_suffixes: Tuple of tag suffixes to filter out

    Returns:
        Filtered list without function words.
    """
    result = []
    for word, tag in chunk_list:
        if not any(tag.endswith(suffix) for suffix in tag_suffixes):
            result.append((word, tag))
    return result


def is_filelike(obj):
    """Check if an object is file-like (has a read method).

    Args:
        obj: The object to check

    Returns:
        True if the object has a read method, False otherwise.
    """
    return hasattr(obj, 'read') and callable(obj.read)


def penn_to_wordnet(tag):
    """Convert a Penn Treebank POS tag to a WordNet POS tag.

    Args:
        tag: Penn Treebank POS tag (e.g., 'NN', 'VB', 'JJ')

    Returns:
        WordNet POS constant or None if no mapping exists.
    """
    from nltk.corpus import wordnet

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def normalize_tags(tagged):
    """Normalize tagged tuples to ensure consistent format.

    Args:
        tagged: List of tagged items (word, tag) or (word, tag, ...)

    Returns:
        List of (word, tag) tuples.
    """
    return [(item[0], item[1]) for item in tagged]
