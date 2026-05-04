"""Tests for the Blobber class."""

import pytest

from textblob import Blobber, TextBlob
from textblob.sentiments import PatternAnalyzer
from textblob.taggers import NLTKTagger
from textblob.tokenizers import WordTokenizer


class TestBlobberCreation:

    def test_create_blobber_default(self):
        """Test creating a Blobber with default settings."""
        tb = Blobber()
        assert tb.tokenizer is not None
        assert tb.pos_tagger is not None
        assert tb.analyzer is not None

    def test_create_blobber_with_custom_tokenizer(self):
        """Test creating a Blobber with custom tokenizer."""
        tokenizer = WordTokenizer()
        tb = Blobber(tokenizer=tokenizer)
        assert tb.tokenizer is tokenizer

    def test_create_blobber_with_custom_pos_tagger(self):
        """Test creating a Blobber with custom POS tagger."""
        tagger = NLTKTagger()
        tb = Blobber(pos_tagger=tagger)
        assert tb.pos_tagger is tagger

    def test_create_blobber_with_custom_analyzer(self):
        """Test creating a Blobber with custom analyzer."""
        analyzer = PatternAnalyzer()
        tb = Blobber(analyzer=analyzer)
        assert tb.analyzer is analyzer


class TestBlobberCall:

    def test_call_creates_textblob(self):
        """Test that calling Blobber creates a TextBlob."""
        tb = Blobber()
        blob = tb("Hello world.")
        assert isinstance(blob, TextBlob)
        assert blob.raw == "Hello world."

    def test_call_uses_blobber_settings(self):
        """Test that Blobber passes its settings to TextBlob."""
        tokenizer = WordTokenizer()
        tagger = NLTKTagger()
        analyzer = PatternAnalyzer()
        tb = Blobber(tokenizer=tokenizer, pos_tagger=tagger, analyzer=analyzer)
        blob = tb("Hello world.")
        assert blob.tokenizer is tokenizer
        assert blob.pos_tagger is tagger
        assert blob.analyzer is analyzer

    def test_multiple_blobs_share_settings(self):
        """Test that multiple blobs from same Blobber share settings."""
        tb = Blobber()
        blob1 = tb("Hello world.")
        blob2 = tb("Beautiful is better than ugly.")
        assert blob1.tokenizer is blob2.tokenizer
        assert blob1.pos_tagger is blob2.pos_tagger
        assert blob1.analyzer is blob2.analyzer


class TestBlobberRepr:

    def test_repr(self):
        """Test Blobber repr."""
        tb = Blobber()
        repr_str = repr(tb)
        assert "Blobber" in repr_str
        assert "tokenizer" in repr_str
        assert "pos_tagger" in repr_str
        assert "analyzer" in repr_str

    def test_repr_shows_class_names(self):
        """Test that repr shows class names."""
        tb = Blobber()
        repr_str = repr(tb)
        assert "WordTokenizer" in repr_str
        assert "NLTKTagger" in repr_str
        assert "PatternAnalyzer" in repr_str


class TestBlobberUsage:

    def test_blobber_workflow(self):
        """Test typical Blobber workflow."""
        tb = Blobber()

        # Create multiple blobs
        texts = ["Hello world.", "How are you?", "Beautiful is better than ugly."]
        blobs = [tb(text) for text in texts]

        # All blobs should work correctly
        assert len(blobs) == 3
        for blob, text in zip(blobs, texts):
            assert blob.raw == text
            assert len(blob.words) > 0


if __name__ == "__main__":
    pytest.main([__file__])
