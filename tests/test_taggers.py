"""Tests for taggers."""

import pytest

from textblob.taggers import NLTKTagger


class TestNLTKTagger:

    def setup_method(self):
        self.tagger = NLTKTagger()
        self.text = "Simple is better than complex."

    def test_tag_returns_list_of_tuples(self):
        """Test that tag() returns a list of (word, tag) tuples."""
        result = self.tagger.tag(self.text)
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) for item in result)
        assert all(len(item) == 2 for item in result)

    def test_tag_uses_penn_treebank_tagset(self):
        """Test that tags use Penn Treebank tagset."""
        result = self.tagger.tag(self.text)
        # Check that common tags are present
        tags = [tag for _, tag in result]
        # 'is' should be tagged as VBZ (verb, 3rd person singular present)
        is_tag = [tag for word, tag in result if word.lower() == "is"]
        assert "VBZ" in is_tag

    def test_tag_without_tokenize(self):
        """Test tagging with tokenize=False splits on whitespace."""
        result = self.tagger.tag("Hello world", tokenize=False)
        assert len(result) == 2

    def test_tag_with_blob_like_object(self):
        """Test tagging with an object that has a tokens attribute."""

        class MockBlob:
            tokens = ["Hello", "world", "!"]

        mock = MockBlob()
        result = self.tagger.tag(mock)
        assert len(result) == 3

    def test_tag_sentence(self):
        """Test tagging a simple sentence."""
        result = self.tagger.tag("The quick brown fox jumps.")
        words = [word for word, tag in result]
        assert "The" in words
        assert "quick" in words
        assert "fox" in words


if __name__ == "__main__":
    pytest.main([__file__])
