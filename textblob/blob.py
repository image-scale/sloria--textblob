"""Main TextBlob classes: Word, WordList, BaseBlob, TextBlob, Sentence, Blobber."""

import json
from collections import defaultdict

from .decorators import cached_property
from .mixins import BlobComparableMixin, StringlikeMixin
from .inflect import singularize, pluralize
from .spelling import correct as spell_correct, spellcheck
from .utils import strip_punc


class Word(str):
    """A simple word representation with NLP capabilities.

    Inherits from str, so it can be used anywhere a string is expected.
    """

    def __new__(cls, string, pos_tag=None):
        return str.__new__(cls, string)

    def __init__(self, string, pos_tag=None):
        self.pos_tag = pos_tag

    def __repr__(self):
        return f"Word('{self}')"

    def singularize(self):
        """Return the singular form of the word."""
        return Word(singularize(self))

    def pluralize(self):
        """Return the plural form of the word."""
        return Word(pluralize(self))

    def lemmatize(self, pos=None):
        """Return the lemma (base form) of the word.

        Args:
            pos: Part of speech ('n' for noun, 'v' for verb, etc.)
                If None, defaults to noun.
        """
        try:
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
            if pos is None:
                pos = 'n'
            return lemmatizer.lemmatize(self.lower(), pos)
        except LookupError:
            return self.lower()

    @cached_property
    def lemma(self):
        """The lemma of the word (cached)."""
        return self.lemmatize()

    def stem(self, stemmer=None):
        """Return the stemmed form of the word.

        Args:
            stemmer: An NLTK stemmer instance. Defaults to PorterStemmer.
        """
        if stemmer is None:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
        return stemmer.stem(self)

    def spellcheck(self):
        """Return a list of (word, confidence) spelling suggestions."""
        return spellcheck(self)

    def correct(self):
        """Return the spell-corrected word."""
        return Word(spell_correct(self))

    @cached_property
    def synsets(self):
        """List of WordNet synsets for this word."""
        return self.get_synsets()

    def get_synsets(self, pos=None):
        """Get WordNet synsets for this word.

        Args:
            pos: Part of speech filter (wordnet.NOUN, wordnet.VERB, etc.)
        """
        try:
            from nltk.corpus import wordnet
            if pos:
                return wordnet.synsets(self, pos=pos)
            return wordnet.synsets(self)
        except LookupError:
            return []

    @cached_property
    def definitions(self):
        """List of definitions for this word."""
        return self.define()

    def define(self, pos=None):
        """Get definitions for this word.

        Args:
            pos: Part of speech filter
        """
        synsets = self.get_synsets(pos=pos)
        return [syn.definition() for syn in synsets]


Word.PorterStemmer = None
Word.LancasterStemmer = None
Word.SnowballStemmer = None

try:
    from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
    Word.PorterStemmer = PorterStemmer
    Word.LancasterStemmer = LancasterStemmer
    Word.SnowballStemmer = SnowballStemmer
except ImportError:
    pass


class WordList(list):
    """A list-like collection of Word objects."""

    def __init__(self, collection):
        super().__init__([Word(w) if not isinstance(w, Word) else w
                          for w in collection])

    def __repr__(self):
        return f"WordList({super().__repr__()})"

    def __getitem__(self, key):
        item = super().__getitem__(key)
        if isinstance(key, slice):
            return WordList(item)
        return item

    def __getslice__(self, i, j):
        return self.__getitem__(slice(i, j))

    def count(self, strg, case_sensitive=False):
        """Count occurrences of a string in the list."""
        if case_sensitive:
            return sum(1 for w in self if w == strg)
        else:
            strg_lower = strg.lower()
            return sum(1 for w in self if w.lower() == strg_lower)

    def append(self, obj):
        if not isinstance(obj, Word):
            obj = Word(obj)
        super().append(obj)

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def upper(self):
        """Return a new WordList with all words uppercased."""
        return WordList([w.upper() for w in self])

    def lower(self):
        """Return a new WordList with all words lowercased."""
        return WordList([w.lower() for w in self])

    def singularize(self):
        """Return a new WordList with all words singularized."""
        return WordList([Word(w).singularize() for w in self])

    def pluralize(self):
        """Return a new WordList with all words pluralized."""
        return WordList([Word(w).pluralize() for w in self])

    def lemmatize(self):
        """Return a new WordList with all words lemmatized."""
        return WordList([Word(w).lemmatize() for w in self])

    def stem(self, *args, **kwargs):
        """Return a new WordList with all words stemmed."""
        return WordList([Word(w).stem(*args, **kwargs) for w in self])


class BaseBlob(BlobComparableMixin, StringlikeMixin):
    """Abstract base class for TextBlob and Sentence.

    Provides common text processing functionality.
    """

    def __init__(self, text, tokenizer=None, pos_tagger=None,
                 np_extractor=None, analyzer=None, parser=None,
                 classifier=None):
        if not isinstance(text, str):
            raise TypeError('Text must be a string.')
        if not text.strip():
            self.raw = ''
        else:
            self.raw = text

        self.tokenizer = tokenizer
        self.pos_tagger = pos_tagger
        self.np_extractor = np_extractor
        self.analyzer = analyzer
        self.parser = parser
        self.classifier = classifier

    def __repr__(self):
        class_name = self.__class__.__name__
        text = self.raw[:70] + '...' if len(self.raw) > 70 else self.raw
        return f'{class_name}("{text}")'

    def _get_tokenizer(self):
        if self.tokenizer is None:
            from .tokenizers import WordTokenizer
            self.tokenizer = WordTokenizer()
        return self.tokenizer

    def _get_pos_tagger(self):
        if self.pos_tagger is None:
            from .taggers import NLTKTagger
            self.pos_tagger = NLTKTagger()
        return self.pos_tagger

    def _get_np_extractor(self):
        if self.np_extractor is None:
            from .np_extractors import FastNPExtractor
            self.np_extractor = FastNPExtractor()
        return self.np_extractor

    def _get_analyzer(self):
        if self.analyzer is None:
            from .sentiments import PatternAnalyzer
            self.analyzer = PatternAnalyzer()
        return self.analyzer

    def _get_parser(self):
        if self.parser is None:
            from .parsers import PatternParser
            self.parser = PatternParser()
        return self.parser

    @cached_property
    def words(self):
        """Return a WordList of word tokens (excludes punctuation)."""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.tokenize(self.raw, include_punc=False)
        return WordList(tokens)

    @cached_property
    def tokens(self):
        """Return a WordList of all tokens (including punctuation)."""
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.tokenize(self.raw, include_punc=True)
        return WordList(tokens)

    def tokenize(self, tokenizer=None):
        """Tokenize the text."""
        t = tokenizer or self._get_tokenizer()
        return WordList(t.tokenize(self.raw))

    @cached_property
    def tags(self):
        """Return a list of (word, POS tag) tuples."""
        tagger = self._get_pos_tagger()
        return tagger.tag(self.raw)

    pos_tags = tags

    @cached_property
    def noun_phrases(self):
        """Return a WordList of noun phrases."""
        extractor = self._get_np_extractor()
        return WordList(extractor.extract(self.raw))

    @cached_property
    def sentiment(self):
        """Return a Sentiment named tuple with polarity and subjectivity."""
        analyzer = self._get_analyzer()
        return analyzer.analyze(self.raw)

    @property
    def polarity(self):
        """Return the polarity score (-1.0 to 1.0)."""
        return self.sentiment.polarity

    @property
    def subjectivity(self):
        """Return the subjectivity score (0.0 to 1.0)."""
        return self.sentiment.subjectivity

    @cached_property
    def word_counts(self):
        """Return a dictionary of word frequencies."""
        counts = defaultdict(int)
        for word in self.words:
            counts[word.lower()] += 1
        return counts

    @cached_property
    def np_counts(self):
        """Return a dictionary of noun phrase frequencies."""
        counts = defaultdict(int)
        for np in self.noun_phrases:
            counts[np.lower()] += 1
        return counts

    def ngrams(self, n=3):
        """Return a list of n-grams (WordLists).

        Args:
            n: The length of each n-gram (default 3)
        """
        words = self.words
        if n <= 0 or n > len(words):
            return []
        return [WordList(words[i:i+n]) for i in range(len(words) - n + 1)]

    def parse(self, parser=None):
        """Parse the text and return a parsed string."""
        p = parser or self._get_parser()
        return p.parse(self.raw)

    def classify(self):
        """Classify the text using the blob's classifier."""
        if self.classifier is None:
            raise ValueError('No classifier configured for this blob.')
        return self.classifier.classify(self.raw)

    def correct(self):
        """Return a new blob with spelling corrected."""
        tokens = []
        for token in self.tokens:
            if token.isalpha():
                tokens.append(spell_correct(token))
            else:
                tokens.append(token)
        corrected = ' '.join(tokens)
        corrected = corrected.replace(' ,', ',').replace(' .', '.')
        corrected = corrected.replace(' !', '!').replace(' ?', '?')
        corrected = corrected.replace(" '", "'").replace(" n't", "n't")
        return self.__class__(corrected)


class Sentence(BaseBlob):
    """A single sentence within a TextBlob."""

    def __init__(self, sentence, start_index=0, end_index=None, **kwargs):
        super().__init__(sentence, **kwargs)
        self.start = self.start_index = start_index
        if end_index is None:
            end_index = start_index + len(sentence)
        self.end = self.end_index = end_index

    @property
    def dict(self):
        """Return a dictionary representation of the sentence."""
        return {
            'raw': self.raw,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'polarity': self.polarity,
            'subjectivity': self.subjectivity,
        }


class TextBlob(BaseBlob):
    """Main class for processing text.

    Example:
        >>> blob = TextBlob("Hello world. How are you?")
        >>> blob.words
        WordList(['Hello', 'world', 'How', 'are', 'you'])
        >>> blob.sentences
        [Sentence("Hello world."), Sentence("How are you?")]
    """

    @cached_property
    def sentences(self):
        """Return a list of Sentence objects."""
        from .tokenizers import SentenceTokenizer
        tokenizer = SentenceTokenizer()
        sentence_objects = []
        for start, end in tokenizer.span_tokenize(self.raw):
            sent_text = self.raw[start:end]
            sentence = Sentence(
                sent_text,
                start_index=start,
                end_index=end,
                tokenizer=self.tokenizer,
                pos_tagger=self.pos_tagger,
                np_extractor=self.np_extractor,
                analyzer=self.analyzer,
                parser=self.parser,
                classifier=self.classifier,
            )
            sentence_objects.append(sentence)
        return sentence_objects

    @cached_property
    def raw_sentences(self):
        """Return a list of raw sentence strings."""
        return [sentence.raw for sentence in self.sentences]

    @property
    def serialized(self):
        """Return a list of dict representations of sentences."""
        return [sentence.dict for sentence in self.sentences]

    @property
    def json(self):
        """Return a JSON representation of the blob."""
        return self.to_json()

    def to_json(self, *args, **kwargs):
        """Return a JSON string representation."""
        return json.dumps(self.serialized, *args, **kwargs)


class Blobber:
    """Factory for creating TextBlobs with pre-configured components.

    Useful when creating many TextBlobs that should share the same
    tokenizer, tagger, or analyzer instances.

    Example:
        >>> tb = Blobber(analyzer=NaiveBayesAnalyzer())
        >>> blob1 = tb("This is great!")
        >>> blob2 = tb("This is terrible.")
        >>> blob1.analyzer is blob2.analyzer
        True
    """

    def __init__(self, tokenizer=None, pos_tagger=None, np_extractor=None,
                 analyzer=None, parser=None, classifier=None):
        self.tokenizer = tokenizer
        self.pos_tagger = pos_tagger
        self.np_extractor = np_extractor
        self.analyzer = analyzer
        self.parser = parser
        self.classifier = classifier

    def __call__(self, text):
        """Create a TextBlob with pre-configured components."""
        return TextBlob(
            text,
            tokenizer=self.tokenizer,
            pos_tagger=self.pos_tagger,
            np_extractor=self.np_extractor,
            analyzer=self.analyzer,
            parser=self.parser,
            classifier=self.classifier,
        )

    def __repr__(self):
        parts = []
        if self.tokenizer:
            parts.append(f'tokenizer={self.tokenizer.__class__.__name__}')
        if self.pos_tagger:
            parts.append(f'pos_tagger={self.pos_tagger.__class__.__name__}')
        if self.np_extractor:
            parts.append(f'np_extractor={self.np_extractor.__class__.__name__}')
        if self.analyzer:
            parts.append(f'analyzer={self.analyzer.__class__.__name__}')
        if self.parser:
            parts.append(f'parser={self.parser.__class__.__name__}')
        if self.classifier:
            parts.append(f'classifier={self.classifier.__class__.__name__}')
        return f"Blobber({', '.join(parts)})"
