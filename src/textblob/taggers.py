"""Part-of-speech tagger implementations."""

import nltk

from textblob.base import BaseTagger
from textblob.decorators import requires_nltk_corpus


class NLTKTagger(BaseTagger):
    """Tagger that uses NLTK's standard TreeBank tagger.

    Uses NLTK's pos_tag function which applies the averaged perceptron tagger
    trained on the Penn Treebank.
    """

    @requires_nltk_corpus
    def tag(self, text, tokenize=True):
        """Tag a string or BaseBlob.

        :param text: A string or a BaseBlob-like object with a tokens attribute.
        :param tokenize: If True, tokenize the text first (default).
        :returns: List of (word, tag) tuples.
        """
        if isinstance(text, str):
            if tokenize:
                tokens = nltk.tokenize.word_tokenize(text)
            else:
                tokens = text.split()
        else:
            # Assume it's a blob-like object with tokens attribute
            tokens = text.tokens

        return nltk.tag.pos_tag(tokens)
