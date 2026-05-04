"""Tests for tokenizers."""

import pytest

from textblob.tokenizers import (
    SentenceTokenizer,
    WordTokenizer,
    sent_tokenize,
    word_tokenize,
)


def is_generator(obj):
    return hasattr(obj, "__next__")


class TestWordTokenizer:

    def setup_method(self):
        self.tokenizer = WordTokenizer()
        self.text = "Python is a high-level programming language."

    def test_tokenize(self):
        assert self.tokenizer.tokenize(self.text) == [
            "Python",
            "is",
            "a",
            "high-level",
            "programming",
            "language",
            ".",
        ]

    def test_exclude_punc(self):
        assert self.tokenizer.tokenize(self.text, include_punc=False) == [
            "Python",
            "is",
            "a",
            "high-level",
            "programming",
            "language",
        ]

    def test_itokenize(self):
        gen = self.tokenizer.itokenize(self.text)
        assert next(gen) == "Python"
        assert next(gen) == "is"

    def test_word_tokenize(self):
        tokens = word_tokenize(self.text)
        assert is_generator(tokens)
        assert list(tokens) == self.tokenizer.tokenize(self.text)

    def test_contractions(self):
        text = "I can't believe it's not butter."
        tokens = self.tokenizer.tokenize(text, include_punc=False)
        # Contractions should be split but apostrophe words preserved
        assert "ca" in tokens
        assert "n't" in tokens
        assert "'s" in tokens


class TestSentenceTokenizer:

    def setup_method(self):
        self.tokenizer = SentenceTokenizer()
        self.text = "Beautiful is better than ugly. Simple is better than complex."

    def test_tokenize(self):
        assert self.tokenizer.tokenize(self.text) == [
            "Beautiful is better than ugly.",
            "Simple is better than complex.",
        ]

    def test_itokenize(self):
        gen = self.tokenizer.itokenize(self.text)
        assert next(gen) == "Beautiful is better than ugly."
        assert next(gen) == "Simple is better than complex."

    def test_sent_tokenize(self):
        tokens = sent_tokenize(self.text)
        assert is_generator(tokens)  # It's a generator
        assert list(tokens) == self.tokenizer.tokenize(self.text)

    def test_single_sentence(self):
        text = "Just one sentence"
        result = self.tokenizer.tokenize(text)
        assert len(result) == 1
        assert result[0] == "Just one sentence"


if __name__ == "__main__":
    pytest.main([__file__])
