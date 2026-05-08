"""Part-of-speech taggers for TextBlob."""

from .base import BaseTagger
from .decorators import requires_nltk_corpus
from .tokenizers import word_tokenize


class NLTKTagger(BaseTagger):
    """POS tagger using NLTK's averaged perceptron tagger.

    Uses the Penn Treebank tagset:
    - NN: noun, singular
    - NNS: noun, plural
    - NNP: proper noun, singular
    - VB: verb, base form
    - VBD: verb, past tense
    - JJ: adjective
    - RB: adverb
    - etc.
    """

    def __init__(self):
        self._tagger = None

    @property
    @requires_nltk_corpus('averaged_perceptron_tagger_eng')
    def tagger(self):
        if self._tagger is None:
            import nltk
            self._tagger = nltk.pos_tag
        return self._tagger

    def tag(self, text, tokenize=True):
        """Tag a string with part-of-speech tags.

        Args:
            text: The text to tag (string or list of tokens)
            tokenize: If True, tokenize the text first.
                     If False, text must be a list of tokens.

        Returns:
            List of (word, tag) tuples.
        """
        if tokenize:
            tokens = word_tokenize(text, include_punc=True)
        else:
            tokens = text
        return self.tagger(tokens)


class PatternTagger(BaseTagger):
    """POS tagger using a Pattern-like lexicon-based approach.

    This tagger uses a combination of:
    - A word/POS lexicon
    - Morphological rules for unknown words
    - Contextual rules for disambiguation
    """

    def __init__(self):
        self._lexicon = None

    @property
    def lexicon(self):
        if self._lexicon is None:
            self._lexicon = _load_lexicon()
        return self._lexicon

    def tag(self, text, tokenize=True):
        """Tag a string with part-of-speech tags.

        Args:
            text: The text to tag
            tokenize: If True, tokenize the text first

        Returns:
            List of (word, tag) tuples.
        """
        if tokenize:
            tokens = word_tokenize(text, include_punc=True)
        else:
            tokens = text

        tagged = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token in self.lexicon:
                tag = self.lexicon[lower_token]
            else:
                tag = self._guess_tag(token)
            tagged.append((token, tag))

        return self._apply_context_rules(tagged)

    def _guess_tag(self, word):
        """Guess the POS tag for an unknown word based on morphology."""
        lower = word.lower()

        if word[0].isupper():
            return 'NNP'
        if lower.endswith('ing'):
            return 'VBG'
        if lower.endswith('ed'):
            return 'VBD'
        if lower.endswith('ly'):
            return 'RB'
        if lower.endswith('ness') or lower.endswith('ment') or lower.endswith('tion'):
            return 'NN'
        if lower.endswith('able') or lower.endswith('ible') or lower.endswith('ful'):
            return 'JJ'
        if lower.endswith('s') and len(lower) > 2:
            return 'NNS'

        return 'NN'

    def _apply_context_rules(self, tagged):
        """Apply contextual rules to improve tagging accuracy."""
        result = list(tagged)

        for i, (word, tag) in enumerate(result):
            if i > 0:
                prev_word, prev_tag = result[i - 1]
                if prev_tag == 'DT' and tag.startswith('VB'):
                    result[i] = (word, 'NN')
                elif prev_tag in ('MD', 'TO') and not tag.startswith('VB'):
                    result[i] = (word, 'VB')

        return result


def _load_lexicon():
    """Load a basic POS lexicon."""
    lexicon = {
        'the': 'DT', 'a': 'DT', 'an': 'DT',
        'is': 'VBZ', 'are': 'VBP', 'was': 'VBD', 'were': 'VBD',
        'be': 'VB', 'been': 'VBN', 'being': 'VBG',
        'have': 'VBP', 'has': 'VBZ', 'had': 'VBD',
        'do': 'VBP', 'does': 'VBZ', 'did': 'VBD', 'done': 'VBN',
        'will': 'MD', 'would': 'MD', 'could': 'MD', 'should': 'MD',
        'may': 'MD', 'might': 'MD', 'must': 'MD', 'can': 'MD',
        'i': 'PRP', 'you': 'PRP', 'he': 'PRP', 'she': 'PRP',
        'it': 'PRP', 'we': 'PRP', 'they': 'PRP',
        'this': 'DT', 'that': 'DT', 'these': 'DT', 'those': 'DT',
        'and': 'CC', 'or': 'CC', 'but': 'CC',
        'in': 'IN', 'on': 'IN', 'at': 'IN', 'to': 'TO', 'for': 'IN',
        'with': 'IN', 'by': 'IN', 'from': 'IN', 'of': 'IN',
        'not': 'RB', 'very': 'RB', 'also': 'RB',
        'good': 'JJ', 'bad': 'JJ', 'great': 'JJ', 'nice': 'JJ',
        'new': 'JJ', 'old': 'JJ', 'big': 'JJ', 'small': 'JJ',
        '.': '.', ',': ',', '!': '.', '?': '.', ':': ':', ';': ':',
        "'s": 'POS', "n't": 'RB',
    }
    return lexicon
