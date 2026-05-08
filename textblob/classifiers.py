"""Text classifiers for TextBlob."""

import json
import csv

from .utils import is_filelike


def basic_extractor(document, train_set):
    """Extract word presence features from a document.

    Args:
        document: The text to extract features from
        train_set: The training set (used to determine vocabulary)

    Returns:
        Dict of {word: True} for words present in document
    """
    words = set(document.lower().split())
    return {word: True for word in words}


def contains_extractor(document):
    """Extract word presence features without a training set.

    Args:
        document: The text to extract features from

    Returns:
        Dict of {word: True} for words present in document
    """
    words = set(document.lower().split())
    return {word: True for word in words}


class BaseClassifier:
    """Abstract base class for text classifiers."""

    def __init__(self, train_set, feature_extractor=basic_extractor,
                 format=None, **kwargs):
        """Initialize the classifier.

        Args:
            train_set: Training data as list of (text, label) tuples,
                      or a file-like object
            feature_extractor: Function to extract features from text
            format: File format ('json', 'csv', 'tsv') for file input
        """
        self.feature_extractor = feature_extractor
        self.train_set = self._read_data(train_set, format)
        self.train_features = [
            (self.feature_extractor(text, self.train_set), label)
            for text, label in self.train_set
        ]

    def _read_data(self, data, format=None):
        """Read training data from various sources."""
        if isinstance(data, list):
            return data

        if is_filelike(data):
            return self._read_file(data, format)

        return data

    def _read_file(self, fp, format=None):
        """Read training data from a file."""
        if format is None:
            format = 'csv'

        content = fp.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        if format == 'json':
            data = json.loads(content)
            return [(item['text'], item['label']) for item in data]

        elif format == 'tsv':
            lines = content.strip().split('\n')
            return [tuple(line.split('\t', 1)[::-1]) for line in lines if line]

        else:
            lines = content.strip().split('\n')
            result = []
            for line in lines:
                if ',' in line:
                    parts = line.rsplit(',', 1)
                    if len(parts) == 2:
                        result.append((parts[0].strip('"'), parts[1].strip()))
            return result

    def classify(self, text):
        """Classify a piece of text."""
        raise NotImplementedError

    def accuracy(self, test_set):
        """Compute accuracy on a test set.

        Args:
            test_set: List of (text, label) tuples

        Returns:
            Accuracy as a float between 0 and 1
        """
        correct = 0
        for text, label in test_set:
            if self.classify(text) == label:
                correct += 1
        return correct / len(test_set) if test_set else 0.0


class NaiveBayesClassifier(BaseClassifier):
    """Naive Bayes text classifier."""

    def __init__(self, train_set, feature_extractor=basic_extractor, **kwargs):
        super().__init__(train_set, feature_extractor, **kwargs)
        self._classifier = self._train()

    def _train(self):
        """Train the classifier."""
        from nltk.classify import NaiveBayesClassifier
        return NaiveBayesClassifier.train(self.train_features)

    def classify(self, text):
        """Classify a piece of text.

        Args:
            text: The text to classify

        Returns:
            The predicted label
        """
        features = self.feature_extractor(text, self.train_set)
        return self._classifier.classify(features)

    def prob_classify(self, text):
        """Return probability distribution over labels.

        Args:
            text: The text to classify

        Returns:
            NLTK ProbDistI object
        """
        features = self.feature_extractor(text, self.train_set)
        return self._classifier.prob_classify(features)

    def update(self, new_data):
        """Update the classifier with new training data.

        Args:
            new_data: List of (text, label) tuples
        """
        self.train_set.extend(new_data)
        new_features = [
            (self.feature_extractor(text, self.train_set), label)
            for text, label in new_data
        ]
        self.train_features.extend(new_features)
        self._classifier = self._train()

    def informative_features(self, n=10):
        """Return the most informative features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature, label) tuples
        """
        return self._classifier.most_informative_features(n)

    def show_informative_features(self, n=10):
        """Print the most informative features."""
        self._classifier.show_most_informative_features(n)


class DecisionTreeClassifier(BaseClassifier):
    """Decision tree text classifier."""

    def __init__(self, train_set, feature_extractor=basic_extractor, **kwargs):
        super().__init__(train_set, feature_extractor, **kwargs)
        self._classifier = self._train()

    def _train(self):
        """Train the classifier."""
        from nltk.classify import DecisionTreeClassifier as NLTKDecisionTree
        return NLTKDecisionTree.train(self.train_features)

    def classify(self, text):
        """Classify a piece of text."""
        features = self.feature_extractor(text, self.train_set)
        return self._classifier.classify(features)

    def pretty_format(self):
        """Return a string representation of the decision tree."""
        return self._classifier.pretty_format()

    def pprint(self):
        """Print the decision tree."""
        print(self.pretty_format())

    def pseudocode(self):
        """Return pseudocode representation of the decision rules."""
        return self._classifier.pseudocode()


class PositiveNaiveBayesClassifier(BaseClassifier):
    """Semi-supervised Naive Bayes classifier for positive-only training.

    Uses positive examples and unlabeled data to train.
    """

    def __init__(self, positive_set, unlabeled_set,
                 feature_extractor=basic_extractor, **kwargs):
        self.feature_extractor = feature_extractor
        self.positive_set = positive_set
        self.unlabeled_set = unlabeled_set
        self.train_set = positive_set + unlabeled_set
        self._classifier = self._train()

    def _train(self):
        """Train using positive and unlabeled data."""
        from nltk.classify import PositiveNaiveBayesClassifier as PNBC

        positive_features = [
            self.feature_extractor(text, self.train_set)
            for text, _ in self.positive_set
        ]
        unlabeled_features = [
            self.feature_extractor(text, self.train_set)
            for text in self.unlabeled_set
        ]

        return PNBC.train(positive_features, unlabeled_features)

    def classify(self, text):
        """Classify a piece of text."""
        features = self.feature_extractor(text, self.train_set)
        return self._classifier.classify(features)


class MaxEntClassifier(BaseClassifier):
    """Maximum Entropy text classifier."""

    def __init__(self, train_set, feature_extractor=basic_extractor, **kwargs):
        super().__init__(train_set, feature_extractor, **kwargs)
        self._classifier = self._train()

    def _train(self):
        """Train the classifier."""
        from nltk.classify import MaxentClassifier
        return MaxentClassifier.train(self.train_features, max_iter=10)

    def classify(self, text):
        """Classify a piece of text."""
        features = self.feature_extractor(text, self.train_set)
        return self._classifier.classify(features)

    def prob_classify(self, text):
        """Return probability distribution over labels."""
        features = self.feature_extractor(text, self.train_set)
        return self._classifier.prob_classify(features)
