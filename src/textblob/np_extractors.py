"""Noun phrase extractors for TextBlob.

Provides extractors for identifying noun phrases in text.
"""

import nltk

from textblob.base import BaseNPExtractor
from textblob.decorators import requires_nltk_corpus


class ConllExtractor(BaseNPExtractor):
    """A noun phrase extractor using NLTK's chunking with a simple
    grammar based on POS tags.

    Uses a regex grammar to identify noun phrases:
    NP: {<DT>?<JJ>*<NN.*>+}
    """

    # Simple NP grammar: optional determiner, any number of adjectives,
    # one or more nouns
    NP_GRAMMAR = r"""
        NP: {<DT>?<JJ.*>*<NN.*>+}
    """

    def __init__(self):
        self._parser = None

    @property
    def parser(self):
        """Lazily create the chunk parser."""
        if self._parser is None:
            self._parser = nltk.RegexpParser(self.NP_GRAMMAR)
        return self._parser

    @requires_nltk_corpus
    def extract(self, text):
        """Extract noun phrases from text using NLTK chunking.

        :param text: The text to extract noun phrases from. Can be a string
            or a TextBlob/Sentence object.
        :returns: A list of noun phrase strings.
        """
        # Get POS tagged tokens
        if hasattr(text, 'pos_tags'):
            # TextBlob or Sentence object
            tagged = text.pos_tags
        else:
            # Plain string - tokenize and tag
            tokens = nltk.word_tokenize(str(text))
            tagged = nltk.pos_tag(tokens)

        # Parse with chunk grammar
        tree = self.parser.parse(tagged)

        # Extract noun phrases
        noun_phrases = []
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                # Join words in the NP
                phrase = ' '.join(word for word, tag in subtree.leaves())
                # Normalize whitespace and lowercase
                phrase = phrase.lower()
                if phrase:
                    noun_phrases.append(phrase)

        return noun_phrases
