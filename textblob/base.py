"""Abstract base classes for TextBlob components."""

from abc import ABC, abstractmethod


class BaseTagger(ABC):
    """Abstract base class for POS taggers.

    Subclasses must implement the `tag` method.
    """

    @abstractmethod
    def tag(self, text, tokenize=True):
        """Tag a string of text.

        Args:
            text: The text to tag
            tokenize: Whether to tokenize the text first

        Returns:
            List of (word, tag) tuples.
        """
        pass


class BaseNPExtractor(ABC):
    """Abstract base class for noun phrase extractors.

    Subclasses must implement the `extract` method.
    """

    @abstractmethod
    def extract(self, text):
        """Extract noun phrases from text.

        Args:
            text: The text to extract noun phrases from

        Returns:
            List of noun phrase strings.
        """
        pass


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers.

    Subclasses must implement the `tokenize` method.
    """

    @abstractmethod
    def tokenize(self, text):
        """Tokenize a string of text.

        Args:
            text: The text to tokenize

        Returns:
            List of tokens.
        """
        pass

    def itokenize(self, text):
        """Return a generator of tokens.

        Args:
            text: The text to tokenize

        Yields:
            Individual tokens.
        """
        return iter(self.tokenize(text))


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers.

    Subclasses must implement the `analyze` method.
    """

    DISCRETE = 'discrete'
    CONTINUOUS = 'continuous'

    kind = CONTINUOUS

    def train(self):
        """Train the sentiment analyzer.

        Default implementation does nothing. Subclasses may override.
        """
        pass

    @abstractmethod
    def analyze(self, text):
        """Analyze the sentiment of text.

        Args:
            text: The text to analyze

        Returns:
            Sentiment result (format depends on implementation).
        """
        pass


class BaseParser(ABC):
    """Abstract base class for text parsers.

    Subclasses must implement the `parse` method.
    """

    @abstractmethod
    def parse(self, text):
        """Parse a string of text.

        Args:
            text: The text to parse

        Returns:
            Parsed representation of the text.
        """
        pass
