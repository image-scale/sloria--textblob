"""Sentiment analysis implementations."""

from collections import namedtuple

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob.base import CONTINUOUS, DISCRETE, BaseSentimentAnalyzer
from textblob.decorators import requires_nltk_corpus


class PatternAnalyzer(BaseSentimentAnalyzer):
    """Sentiment analyzer that uses NLTK's VADER lexicon-based analyzer.

    Returns results as a named tuple of the form:
    ``Sentiment(polarity, subjectivity)``

    where polarity is between -1 (negative) and 1 (positive),
    and subjectivity is between 0 (objective) and 1 (subjective).
    """

    kind = CONTINUOUS
    RETURN_TYPE = namedtuple("Sentiment", ["polarity", "subjectivity"])

    def __init__(self):
        super().__init__()
        self._analyzer = None

    @requires_nltk_corpus
    def _get_analyzer(self):
        """Lazily initialize the VADER analyzer."""
        if self._analyzer is None:
            self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    def analyze(self, text, keep_assessments=False):
        """Return the sentiment as a named tuple of the form:
        ``Sentiment(polarity, subjectivity)``.

        :param text: The text to analyze.
        :param keep_assessments: Not used, kept for compatibility.
        :returns: Named tuple with polarity and subjectivity scores.
        """
        analyzer = self._get_analyzer()
        scores = analyzer.polarity_scores(str(text))

        # compound is between -1 and 1, use as polarity
        polarity = scores["compound"]

        # Subjectivity: higher positive + negative means more subjective
        # Use (pos + neg) as a measure of subjectivity
        # More neutral = more objective, less neutral = more subjective
        pos_neg_sum = scores["pos"] + scores["neg"]
        # Scale to 0-1 range (max subjectivity when pos+neg = 1)
        subjectivity = min(pos_neg_sum, 1.0)

        return self.RETURN_TYPE(polarity=polarity, subjectivity=subjectivity)


# Aliases for backwards compatibility with base module exports
__all__ = [
    "BaseSentimentAnalyzer",
    "DISCRETE",
    "CONTINUOUS",
    "PatternAnalyzer",
]
