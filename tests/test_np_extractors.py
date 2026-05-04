"""Tests for noun phrase extraction."""

import pytest

from textblob import TextBlob, Sentence, WordList
from textblob.base import BaseNPExtractor
from textblob.np_extractors import ConllExtractor


class TestConllExtractor:

    def test_extractor_inherits_base(self):
        """Test that ConllExtractor inherits from BaseNPExtractor."""
        extractor = ConllExtractor()
        assert isinstance(extractor, BaseNPExtractor)

    def test_extract_returns_list(self):
        """Test that extract returns a list."""
        extractor = ConllExtractor()
        result = extractor.extract("The quick brown fox jumps over the lazy dog.")
        assert isinstance(result, list)

    def test_extract_noun_phrases(self):
        """Test extracting noun phrases from text."""
        extractor = ConllExtractor()
        result = extractor.extract("The quick brown fox jumps over the lazy dog.")
        # Should extract noun phrases like "the quick brown fox", "the lazy dog"
        assert len(result) > 0
        assert all(isinstance(phrase, str) for phrase in result)

    def test_extract_with_textblob(self):
        """Test extracting from a TextBlob object."""
        blob = TextBlob("The happy child plays in the sunny park.")
        extractor = ConllExtractor()
        result = extractor.extract(blob)
        assert len(result) > 0

    def test_noun_phrases_lowercase(self):
        """Test that noun phrases are lowercased."""
        extractor = ConllExtractor()
        result = extractor.extract("The Quick Brown Fox")
        for phrase in result:
            assert phrase == phrase.lower()


class TestTextBlobNounPhrases:

    def test_noun_phrases_property(self):
        """Test TextBlob.noun_phrases property."""
        blob = TextBlob("The quick brown fox jumps over the lazy dog.")
        noun_phrases = blob.noun_phrases
        assert isinstance(noun_phrases, WordList)
        assert len(noun_phrases) > 0

    def test_noun_phrases_content(self):
        """Test that noun_phrases contains expected phrases."""
        blob = TextBlob("The quick brown fox jumps over the lazy dog.")
        noun_phrases = blob.noun_phrases
        # Check for common noun phrases
        phrases_str = " ".join(str(p) for p in noun_phrases)
        assert "fox" in phrases_str or "dog" in phrases_str

    def test_sentence_noun_phrases(self):
        """Test Sentence.noun_phrases property."""
        sent = Sentence("The happy child plays in the sunny park.")
        noun_phrases = sent.noun_phrases
        assert isinstance(noun_phrases, WordList)
        assert len(noun_phrases) > 0

    def test_custom_np_extractor(self):
        """Test passing custom np_extractor to TextBlob."""
        extractor = ConllExtractor()
        blob = TextBlob("The quick brown fox.", np_extractor=extractor)
        assert blob.np_extractor is extractor

    def test_invalid_np_extractor_raises_error(self):
        """Test that invalid np_extractor raises ValueError."""
        with pytest.raises(ValueError):
            TextBlob("Hello", np_extractor="invalid")

    def test_multiple_sentences(self):
        """Test noun phrase extraction with multiple sentences."""
        blob = TextBlob("The big dog runs. A small cat sleeps.")
        noun_phrases = blob.noun_phrases
        assert isinstance(noun_phrases, WordList)


class TestNounPhraseIntegration:

    def test_noun_phrases_in_workflow(self):
        """Test noun phrases in a typical workflow."""
        text = "Natural language processing is a field of computer science. Machine learning helps with text analysis."
        blob = TextBlob(text)

        # Get noun phrases
        nps = blob.noun_phrases

        # Should find some noun phrases
        assert len(nps) > 0

        # WordList should support operations
        assert isinstance(nps, WordList)


if __name__ == "__main__":
    pytest.main([__file__])
