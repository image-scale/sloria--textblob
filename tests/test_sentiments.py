"""Tests for sentiment analysis."""

import pytest

from textblob.base import CONTINUOUS
from textblob.sentiments import PatternAnalyzer


class TestPatternAnalyzer:

    def setup_method(self):
        self.analyzer = PatternAnalyzer()

    def test_kind_is_continuous(self):
        """Test that PatternAnalyzer has kind=CONTINUOUS."""
        assert self.analyzer.kind == CONTINUOUS

    def test_analyze_returns_named_tuple(self):
        """Test that analyze() returns a named tuple with polarity and subjectivity."""
        result = self.analyzer.analyze("I feel great today.")
        assert hasattr(result, "polarity")
        assert hasattr(result, "subjectivity")
        # Named tuple should also be accessible by index
        assert result[0] == result.polarity
        assert result[1] == result.subjectivity

    def test_analyze_positive_text(self):
        """Test that positive text returns positive polarity."""
        result = self.analyzer.analyze("I love this movie. It's wonderful and amazing!")
        assert result.polarity > 0

    def test_analyze_negative_text(self):
        """Test that negative text returns negative polarity."""
        result = self.analyzer.analyze("This is terrible and awful. I hate it.")
        assert result.polarity < 0

    def test_polarity_range(self):
        """Test that polarity is between -1 and 1."""
        positive = self.analyzer.analyze("This is absolutely wonderful and amazing!")
        negative = self.analyzer.analyze("This is absolutely terrible and horrible!")
        neutral = self.analyzer.analyze("This is a table.")

        assert -1.0 <= positive.polarity <= 1.0
        assert -1.0 <= negative.polarity <= 1.0
        assert -1.0 <= neutral.polarity <= 1.0

    def test_subjectivity_range(self):
        """Test that subjectivity is between 0 and 1."""
        result1 = self.analyzer.analyze("I love this!")
        result2 = self.analyzer.analyze("The sky is blue.")
        result3 = self.analyzer.analyze("This is the best thing ever!")

        assert 0.0 <= result1.subjectivity <= 1.0
        assert 0.0 <= result2.subjectivity <= 1.0
        assert 0.0 <= result3.subjectivity <= 1.0

    def test_analyze_empty_text(self):
        """Test that empty text returns neutral values."""
        result = self.analyzer.analyze("")
        assert result.polarity == 0.0
        assert result.subjectivity == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
