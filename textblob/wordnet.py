"""WordNet interface for TextBlob.

Provides convenient access to NLTK's WordNet corpus.
"""

try:
    from nltk.corpus import wordnet

    Synset = wordnet.synset
    Lemma = wordnet.lemma

    VERB = wordnet.VERB
    NOUN = wordnet.NOUN
    ADJ = wordnet.ADJ
    ADV = wordnet.ADV

except ImportError:
    wordnet = None
    Synset = None
    Lemma = None
    VERB = 'v'
    NOUN = 'n'
    ADJ = 'a'
    ADV = 'r'

__all__ = ['wordnet', 'Synset', 'Lemma', 'VERB', 'NOUN', 'ADJ', 'ADV']
