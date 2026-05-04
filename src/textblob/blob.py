"""Wrappers for various units of text, including the main TextBlob class.

Example usage:
    >>> from textblob import TextBlob
    >>> b = TextBlob("Simple is better than complex.")
    >>> b.words
    ['Simple', 'is', 'better', 'than', 'complex']
    >>> b.pos_tags
    [('Simple', 'NN'), ('is', 'VBZ'), ('better', 'JJR'), ('than', 'IN'), ('complex', 'JJ')]
    >>> b.sentiment
    Sentiment(polarity=0.5, subjectivity=0.5)
"""

import sys

import nltk

from textblob.base import BaseSentimentAnalyzer, BaseTagger, BaseTokenizer
from textblob.decorators import cached_property
from textblob.mixins import BlobComparableMixin, StringlikeMixin
from textblob.sentiments import PatternAnalyzer
from textblob.taggers import NLTKTagger
from textblob.tokenizers import WordTokenizer, sent_tokenize, word_tokenize
from textblob.utils import PUNCTUATION_REGEX, lowerstrip
from textblob import inflect as _inflect


class Word(str):
    """A simple word representation. Includes methods for inflection,
    lemmatization, and WordNet integration.

    :param word: A string representing a single word.
    :param pos_tag: (optional) The word's POS tag.

    Example:
        >>> from textblob import Word
        >>> w = Word("dogs")
        >>> w.singularize()
        'dog'
        >>> w = Word("run", pos_tag="VB")
        >>> w.lemmatize()
        'run'
    """

    def __new__(cls, word, pos_tag=None):
        """Create a new Word instance."""
        return super().__new__(cls, word)

    def __init__(self, word, pos_tag=None):
        self.string = word
        self.pos_tag = pos_tag

    def __repr__(self):
        return f"Word('{self}')"

    def singularize(self):
        """Return the singular form of the word as a Word.

        :returns: A Word representing the singular form.
        """
        return Word(_inflect.singularize(self.string))

    def pluralize(self):
        """Return the plural form of the word as a Word.

        :returns: A Word representing the plural form.
        """
        return Word(_inflect.pluralize(self.string))

    def lemmatize(self, pos=None):
        """Return the lemma of the word.

        :param pos: (optional) Part of speech. Can be 'n' (noun), 'v' (verb),
            'a' (adjective), or 'r' (adverb). Defaults to noun.
        :returns: A Word representing the lemma.
        """
        from textblob.wordnet import lemmatize, NOUN
        if pos is None:
            # Try to determine pos from pos_tag if available
            if self.pos_tag:
                pos = _penn_to_wordnet(self.pos_tag)
            else:
                pos = NOUN
        return Word(lemmatize(self.string.lower(), pos))

    def stem(self):
        """Return the stem of the word using the Porter stemmer.

        :returns: A Word representing the stem.
        """
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        return Word(stemmer.stem(self.string))

    def spellcheck(self):
        """Return a list of (word, confidence) tuples representing possible
        spelling corrections.

        :returns: List of (word, confidence) tuples.
        """
        from textblob.spelling import suggest
        suggestions = suggest(self.string)
        # Return Word objects in the tuples
        return [(Word(word), confidence) for word, confidence in suggestions]

    def correct(self):
        """Return the most likely correct spelling of the word.

        :returns: A Word with the corrected spelling.
        """
        from textblob.spelling import correct
        return Word(correct(self.string))

    @property
    def synsets(self):
        """Return a list of WordNet synsets for this word.

        :returns: List of Synset objects.
        """
        from textblob.wordnet import get_synsets
        pos = None
        if self.pos_tag:
            pos = _penn_to_wordnet(self.pos_tag)
        return get_synsets(self.string.lower(), pos)

    @property
    def definitions(self):
        """Return a list of definitions for this word from WordNet.

        :returns: List of definition strings.
        """
        from textblob.wordnet import get_definitions
        pos = None
        if self.pos_tag:
            pos = _penn_to_wordnet(self.pos_tag)
        return get_definitions(self.string.lower(), pos)

    def define(self, pos=None):
        """Return a list of definitions for this word.

        :param pos: (optional) A part of speech tag ('n', 'v', 'a', 'r').
        :returns: List of definition strings.
        """
        from textblob.wordnet import get_definitions
        return get_definitions(self.string.lower(), pos)

    def get_synsets(self, pos=None):
        """Return a list of synsets for this word.

        :param pos: (optional) A part of speech tag ('n', 'v', 'a', 'r').
        :returns: List of Synset objects.
        """
        from textblob.wordnet import get_synsets
        return get_synsets(self.string.lower(), pos)


def _penn_to_wordnet(tag):
    """Convert a Penn Treebank POS tag to a WordNet POS tag.

    :param tag: A Penn Treebank POS tag (e.g., 'NN', 'VB', 'JJ').
    :returns: WordNet POS tag ('n', 'v', 'a', 'r') or None.
    """
    from textblob.wordnet import NOUN, VERB, ADJ, ADV
    if tag.startswith('N'):
        return NOUN
    elif tag.startswith('V'):
        return VERB
    elif tag.startswith('J'):
        return ADJ
    elif tag.startswith('R'):
        return ADV
    return NOUN  # Default to noun


class WordList(list):
    """A list-like collection of Word objects with additional methods for
    bulk operations.

    Example:
        >>> from textblob import WordList
        >>> words = WordList(["cat", "dog", "mouse"])
        >>> words.pluralize()
        WordList(['cats', 'dogs', 'mice'])
        >>> words.upper()
        WordList(['CAT', 'DOG', 'MOUSE'])
    """

    def __init__(self, collection=None):
        """Initialize a WordList.

        :param collection: A list of strings or Words.
        """
        if collection is None:
            collection = []
        # Convert strings to Word objects
        words = [w if isinstance(w, Word) else Word(w) for w in collection]
        super().__init__(words)

    def __repr__(self):
        return f"WordList({super().__repr__()})"

    def __getitem__(self, key):
        """Get item(s) from the WordList."""
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return WordList(result)
        return result

    def count(self, word, case_sensitive=False):
        """Count occurrences of a word in the list.

        :param word: The word to count.
        :param case_sensitive: If False (default), count is case-insensitive.
        :returns: The count of the word.
        """
        if case_sensitive:
            return super().count(word)
        word_lower = word.lower() if hasattr(word, 'lower') else str(word).lower()
        return sum(1 for w in self if str(w).lower() == word_lower)

    def upper(self):
        """Return a new WordList with all words uppercased.

        :returns: A new WordList.
        """
        return WordList([Word(str(w).upper()) for w in self])

    def lower(self):
        """Return a new WordList with all words lowercased.

        :returns: A new WordList.
        """
        return WordList([Word(str(w).lower()) for w in self])

    def singularize(self):
        """Return a new WordList with all words singularized.

        :returns: A new WordList.
        """
        return WordList([w.singularize() if isinstance(w, Word) else Word(w).singularize() for w in self])

    def pluralize(self):
        """Return a new WordList with all words pluralized.

        :returns: A new WordList.
        """
        return WordList([w.pluralize() if isinstance(w, Word) else Word(w).pluralize() for w in self])

    def lemmatize(self, pos=None):
        """Return a new WordList with all words lemmatized.

        :param pos: (optional) Part of speech for lemmatization.
        :returns: A new WordList.
        """
        return WordList([
            w.lemmatize(pos) if isinstance(w, Word) else Word(w).lemmatize(pos)
            for w in self
        ])

    def stem(self):
        """Return a new WordList with all words stemmed.

        :returns: A new WordList.
        """
        return WordList([w.stem() if isinstance(w, Word) else Word(w).stem() for w in self])

    def extend(self, other):
        """Extend the WordList with another iterable.

        :param other: An iterable of words.
        """
        words = [w if isinstance(w, Word) else Word(w) for w in other]
        super().extend(words)

    def append(self, word):
        """Append a word to the WordList.

        :param word: A word to append.
        """
        if not isinstance(word, Word):
            word = Word(word)
        super().append(word)


def _validated_param(obj, name, base_class, default, base_class_name=None):
    """Validates a parameter passed to __init__. Makes sure that obj is
    the correct class. Return obj if it's not None or falls back to default.

    :param obj: The object passed in.
    :param name: The name of the parameter.
    :param base_class: The class that obj must inherit from.
    :param default: The default object to fall back upon if obj is None.
    """
    base_class_name = base_class_name if base_class_name else base_class.__name__
    if obj is not None and not isinstance(obj, base_class):
        raise ValueError(f"{name} must be an instance of {base_class_name}")
    return obj or default


class TextBlob(StringlikeMixin, BlobComparableMixin):
    """A general text block, meant for larger bodies of text.
    Provides methods for text analysis including tokenization, POS tagging,
    sentiment analysis, and more.

    :param str text: A string.
    :param tokenizer: (optional) A tokenizer instance. If ``None``,
        defaults to WordTokenizer().
    :param pos_tagger: (optional) A tagger instance. If ``None``,
        defaults to NLTKTagger().
    :param analyzer: (optional) A sentiment analyzer instance. If ``None``,
        defaults to PatternAnalyzer().

    Example:
        >>> blob = TextBlob("Beautiful is better than ugly.")
        >>> blob.words
        ['Beautiful', 'is', 'better', 'than', 'ugly']
        >>> blob.pos_tags
        [('Beautiful', 'JJ'), ('is', 'VBZ'), ('better', 'JJR'), ('than', 'IN'), ('ugly', 'JJ')]
        >>> blob.sentiment.polarity
        0.5
    """

    tokenizer = WordTokenizer()
    pos_tagger = NLTKTagger()
    analyzer = PatternAnalyzer()

    def __init__(self, text, tokenizer=None, pos_tagger=None, analyzer=None):
        if not isinstance(text, (str, bytes)):
            raise TypeError(
                "The `text` argument passed to `__init__(text)` "
                f"must be a string, not {type(text)}"
            )
        self.raw = self.string = text
        self.stripped = lowerstrip(self.raw, all=True)

        # Validate and set tokenizer
        # tokenizer may be a textblob or an NLTK tokenizer
        self.tokenizer = _validated_param(
            tokenizer,
            "tokenizer",
            base_class=(BaseTokenizer, nltk.tokenize.api.TokenizerI),
            default=TextBlob.tokenizer,
            base_class_name="BaseTokenizer",
        )

        # Validate and set pos_tagger
        self.pos_tagger = _validated_param(
            pos_tagger,
            "pos_tagger",
            base_class=BaseTagger,
            default=TextBlob.pos_tagger,
        )

        # Validate and set analyzer
        self.analyzer = _validated_param(
            analyzer,
            "analyzer",
            base_class=BaseSentimentAnalyzer,
            default=TextBlob.analyzer,
        )

    @cached_property
    def words(self):
        """Return a WordList of Word objects. This excludes punctuation characters.
        If you want to include punctuation characters, access the ``tokens``
        property.

        :returns: A WordList of Word objects.
        """
        return WordList([Word(w) for w in word_tokenize(self.raw, include_punc=False)])

    @cached_property
    def tokens(self):
        """Return a list of tokens, using this blob's tokenizer object
        (defaults to WordTokenizer).
        """
        return self.tokenizer.tokenize(self.raw)

    def tokenize(self, tokenizer=None):
        """Return a list of tokens, using ``tokenizer``.

        :param tokenizer: (optional) A tokenizer object. If None, defaults to
            this blob's default tokenizer.
        """
        t = tokenizer if tokenizer is not None else self.tokenizer
        return t.tokenize(self.raw)

    @cached_property
    def sentences(self):
        """Return list of Sentence objects."""
        return self._create_sentence_objects()

    @property
    def raw_sentences(self):
        """List of strings, the raw sentences in the blob."""
        return [sentence.raw for sentence in self.sentences]

    @cached_property
    def pos_tags(self):
        """Returns a list of tuples of the form (word, POS tag).

        Example:
            [('Beautiful', 'JJ'), ('is', 'VBZ'), ('better', 'JJR')]

        :rtype: list of tuples
        """
        if isinstance(self, TextBlob) and not isinstance(self, Sentence):
            # For TextBlob, aggregate tags from all sentences
            return [
                val
                for sublist in [s.pos_tags for s in self.sentences]
                for val in sublist
            ]
        else:
            # For Sentence or other subclasses
            return [
                (word, str(tag))
                for word, tag in self.pos_tagger.tag(self)
                if not PUNCTUATION_REGEX.match(str(tag))
            ]

    # Alias for pos_tags
    tags = pos_tags

    @cached_property
    def sentiment(self):
        """Return a tuple of form (polarity, subjectivity) where polarity
        is a float within the range [-1.0, 1.0] and subjectivity is a float
        within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is
        very subjective.

        :rtype: named tuple of form ``Sentiment(polarity, subjectivity)``
        """
        return self.analyzer.analyze(self.raw)

    @property
    def polarity(self):
        """Return the polarity score as a float within the range [-1.0, 1.0].

        :rtype: float
        """
        return self.sentiment.polarity

    @property
    def subjectivity(self):
        """Return the subjectivity score as a float within the range [0.0, 1.0]
        where 0.0 is very objective and 1.0 is very subjective.

        :rtype: float
        """
        return self.sentiment.subjectivity

    def _create_sentence_objects(self):
        """Returns a list of Sentence objects from the raw text."""
        sentence_objects = []
        sentences = list(sent_tokenize(self.raw))
        char_index = 0  # Keeps track of character index within the blob
        for sent in sentences:
            # Compute the start and end indices of the sentence
            # within the blob
            start_index = self.raw.index(sent, char_index)
            char_index += len(sent)
            end_index = start_index + len(sent)
            # Create a Sentence object with same settings
            s = Sentence(
                sent,
                start_index=start_index,
                end_index=end_index,
                tokenizer=self.tokenizer,
                pos_tagger=self.pos_tagger,
                analyzer=self.analyzer,
            )
            sentence_objects.append(s)
        return sentence_objects

    def ngrams(self, n=3):
        """Return a list of n-grams (tuples of n successive words) for this
        blob.

        :rtype: List of tuples
        """
        if n <= 0:
            return []
        grams = [
            tuple(self.words[i : i + n]) for i in range(len(self.words) - n + 1)
        ]
        return grams

    def _cmpkey(self):
        """Key used by ComparableMixin to implement all rich comparison
        operators.
        """
        return self.raw

    def _strkey(self):
        """Key used by StringlikeMixin to implement string methods."""
        return self.raw

    def __hash__(self):
        return hash(self._cmpkey())

    def __add__(self, other):
        """Concatenates two text objects the same way Python strings are
        concatenated.

        Arguments:
        - `other`: a string or a text object
        """
        if isinstance(other, (str, bytes)):
            return self.__class__(self.raw + other)
        elif isinstance(other, TextBlob):
            return self.__class__(self.raw + other.raw)
        else:
            raise TypeError(
                f"Operands must be either strings or {self.__class__.__name__} objects"
            )

    def split(self, sep=None, maxsplit=sys.maxsize):
        """Behaves like the built-in str.split() except returns a
        list of strings.

        :rtype: list
        """
        return self._strkey().split(sep, maxsplit)

    @property
    def serialized(self):
        """Return a dictionary representation of this TextBlob.

        :returns: A dict with 'raw', 'sentences', 'polarity', and 'subjectivity'.
        """
        return {
            "raw": self.raw,
            "sentences": [s.dict for s in self.sentences],
            "polarity": self.polarity,
            "subjectivity": self.subjectivity,
        }

    def to_json(self, *args, **kwargs):
        """Return a JSON string representation of this TextBlob.

        :param args: Arguments to pass to json.dumps.
        :param kwargs: Keyword arguments to pass to json.dumps.
        :returns: A JSON string.
        """
        import json
        return json.dumps(self.serialized, *args, **kwargs)


class Sentence(TextBlob):
    """A sentence within a TextBlob. Inherits from TextBlob.

    :param sentence: A string, the raw sentence.
    :param start_index: An int, the index where this sentence begins
                        in a TextBlob. If not given, defaults to 0.
    :param end_index: An int, the index where this sentence ends in
                        a TextBlob. If not given, defaults to the
                        length of the sentence.
    """

    def __init__(self, sentence, start_index=0, end_index=None, *args, **kwargs):
        super().__init__(sentence, *args, **kwargs)
        # The start index within a TextBlob
        self.start = self.start_index = start_index
        # The end index within a TextBlob
        self.end = self.end_index = end_index if end_index is not None else len(sentence)

    def __repr__(self):
        return f'Sentence("{self.raw}")'

    @property
    def dict(self):
        """Return a dictionary representation of this Sentence.

        :returns: A dict with 'raw', 'start', 'end', 'polarity', and 'subjectivity'.
        """
        return {
            "raw": self.raw,
            "start": self.start,
            "end": self.end,
            "polarity": self.polarity,
            "subjectivity": self.subjectivity,
        }

    def json(self, *args, **kwargs):
        """Return a JSON string representation of this Sentence.

        :param args: Arguments to pass to json.dumps.
        :param kwargs: Keyword arguments to pass to json.dumps.
        :returns: A JSON string.
        """
        import json
        return json.dumps(self.dict, *args, **kwargs)
