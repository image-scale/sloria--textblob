"""WordNet integration for TextBlob.

Provides access to WordNet synsets, definitions, and lemmatization.
"""

import nltk

from textblob.decorators import requires_nltk_corpus

# WordNet POS tag mappings
NOUN = "n"
VERB = "v"
ADJ = "a"
ADV = "r"


def _get_wordnet():
    """Lazily import wordnet to avoid import-time NLTK data access."""
    from nltk.corpus import wordnet
    return wordnet


@requires_nltk_corpus
def get_synsets(word, pos=None):
    """Return a list of WordNet synsets for a word.

    :param word: The word to look up.
    :param pos: (optional) Part of speech. One of NOUN, VERB, ADJ, ADV.
    :returns: List of Synset objects.
    """
    wordnet = _get_wordnet()
    if pos is not None:
        return wordnet.synsets(word, pos)
    return wordnet.synsets(word)


@requires_nltk_corpus
def get_definitions(word, pos=None):
    """Return a list of definitions for a word.

    :param word: The word to look up.
    :param pos: (optional) Part of speech.
    :returns: List of definition strings.
    """
    synsets = get_synsets(word, pos)
    return [synset.definition() for synset in synsets]


@requires_nltk_corpus
def lemmatize(word, pos=NOUN):
    """Return the lemma form of a word.

    :param word: The word to lemmatize.
    :param pos: Part of speech. Default is NOUN.
    :returns: Lemmatized word string.
    """
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word, pos)
