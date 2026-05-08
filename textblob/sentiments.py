"""Sentiment analysis for TextBlob."""

from collections import namedtuple

from .base import BaseSentimentAnalyzer
from .tokenizers import word_tokenize


Sentiment = namedtuple('Sentiment', ['polarity', 'subjectivity'])
SentimentAssessment = namedtuple('Sentiment', ['polarity', 'subjectivity', 'assessments'])


class PatternAnalyzer(BaseSentimentAnalyzer):
    """Lexicon-based sentiment analyzer.

    Uses a sentiment lexicon to compute polarity and subjectivity scores.
    - polarity: -1.0 (negative) to 1.0 (positive)
    - subjectivity: 0.0 (objective) to 1.0 (subjective)
    """

    kind = BaseSentimentAnalyzer.CONTINUOUS

    def __init__(self):
        self._lexicon = None

    @property
    def lexicon(self):
        if self._lexicon is None:
            from .en import SENTIMENT_LEXICON
            self._lexicon = SENTIMENT_LEXICON
        return self._lexicon

    def analyze(self, text, keep_assessments=False):
        """Analyze the sentiment of text.

        Args:
            text: The text to analyze
            keep_assessments: If True, include word-level assessments

        Returns:
            Sentiment namedtuple with polarity and subjectivity
        """
        words = word_tokenize(text, include_punc=False)

        polarity_sum = 0.0
        subjectivity_sum = 0.0
        count = 0
        assessments = []

        negation_words = {'not', "n't", 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere'}
        negation = False

        for i, word in enumerate(words):
            lower_word = word.lower()

            if lower_word in negation_words:
                negation = True
                continue

            if lower_word in self.lexicon:
                pol, subj = self.lexicon[lower_word]

                if negation:
                    pol = -pol
                    negation = False

                polarity_sum += pol
                subjectivity_sum += subj
                count += 1

                if keep_assessments:
                    assessments.append((word, pol, subj))
            else:
                negation = False

        if count > 0:
            polarity = polarity_sum / count
            subjectivity = subjectivity_sum / count
        else:
            polarity = 0.0
            subjectivity = 0.0

        polarity = max(-1.0, min(1.0, polarity))
        subjectivity = max(0.0, min(1.0, subjectivity))

        if keep_assessments:
            return SentimentAssessment(polarity, subjectivity, assessments)
        return Sentiment(polarity, subjectivity)


class NaiveBayesAnalyzer(BaseSentimentAnalyzer):
    """Naive Bayes sentiment classifier trained on movie reviews.

    Returns discrete sentiment classification (positive/negative)
    along with probability scores.
    """

    kind = BaseSentimentAnalyzer.DISCRETE

    def __init__(self):
        self._classifier = None

    def train(self):
        """Train the classifier on movie reviews corpus."""
        import nltk
        from nltk.corpus import movie_reviews
        from nltk.classify import NaiveBayesClassifier

        def word_features(words):
            return {word: True for word in words}

        negids = movie_reviews.fileids('neg')
        posids = movie_reviews.fileids('pos')

        negfeats = [(word_features(movie_reviews.words(fileids=[f])), 'neg')
                    for f in negids]
        posfeats = [(word_features(movie_reviews.words(fileids=[f])), 'pos')
                    for f in posids]

        train_set = negfeats + posfeats
        self._classifier = NaiveBayesClassifier.train(train_set)

    @property
    def classifier(self):
        if self._classifier is None:
            self.train()
        return self._classifier

    def analyze(self, text):
        """Analyze the sentiment of text.

        Args:
            text: The text to analyze

        Returns:
            Sentiment namedtuple with classification, p_pos, and p_neg
        """
        words = word_tokenize(text, include_punc=False)
        features = {word.lower(): True for word in words}

        prob_dist = self.classifier.prob_classify(features)
        classification = prob_dist.max()
        p_pos = prob_dist.prob('pos')
        p_neg = prob_dist.prob('neg')

        NaiveBayesSentiment = namedtuple('Sentiment', ['classification', 'p_pos', 'p_neg'])
        return NaiveBayesSentiment(classification, p_pos, p_neg)
