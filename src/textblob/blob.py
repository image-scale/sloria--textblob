"""Wrappers for various units of text, including the main TextBlob class.

Example usage:
    >>> from textblob import TextBlob
    >>> b = TextBlob("Simple is better than complex.")
    >>> b.words
    ['Simple', 'is', 'better', 'than', 'complex']
"""

import sys

from textblob.decorators import cached_property
from textblob.mixins import BlobComparableMixin, StringlikeMixin
from textblob.tokenizers import WordTokenizer, sent_tokenize, word_tokenize
from textblob.utils import lowerstrip


class TextBlob(StringlikeMixin, BlobComparableMixin):
    """A general text block, meant for larger bodies of text.
    Provides methods for text analysis including tokenization, n-grams, and more.

    :param str text: A string.
    :param tokenizer: (optional) A tokenizer instance. If ``None``,
        defaults to WordTokenizer().

    Example:
        >>> blob = TextBlob("Beautiful is better than ugly.")
        >>> blob.words
        ['Beautiful', 'is', 'better', 'than', 'ugly']
        >>> blob.sentences
        [Sentence("Beautiful is better than ugly.")]
    """

    tokenizer = WordTokenizer()

    def __init__(self, text, tokenizer=None):
        if not isinstance(text, (str, bytes)):
            raise TypeError(
                "The `text` argument passed to `__init__(text)` "
                f"must be a string, not {type(text)}"
            )
        self.raw = self.string = text
        self.stripped = lowerstrip(self.raw, all=True)
        if tokenizer is not None:
            self.tokenizer = tokenizer

    @cached_property
    def words(self):
        """Return a list of word tokens. This excludes punctuation characters.
        If you want to include punctuation characters, access the ``tokens``
        property.

        :returns: A list of word tokens.
        """
        return list(word_tokenize(self.raw, include_punc=False))

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
            # Create a Sentence object
            s = Sentence(
                sent,
                start_index=start_index,
                end_index=end_index,
                tokenizer=self.tokenizer,
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
