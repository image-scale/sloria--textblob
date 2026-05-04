"""Tests for the TextBlob class."""

import pytest

from textblob import Sentence, TextBlob
from textblob.taggers import NLTKTagger
from textblob.tokenizers import WordTokenizer


class TestTextBlobCreation:

    def test_init_with_string(self):
        blob = TextBlob("Hello world")
        assert blob.raw == "Hello world"
        assert blob.string == "Hello world"

    def test_init_with_non_string_raises_error(self):
        with pytest.raises(TypeError):
            TextBlob(["invalid"])

        with pytest.raises(TypeError):
            TextBlob(12345)

        with pytest.raises(TypeError):
            TextBlob(None)

    def test_raw_attribute(self):
        text = "Some sample text"
        blob = TextBlob(text)
        assert blob.raw == text

    def test_stripped_attribute(self):
        blob = TextBlob("Hello, World!")
        assert blob.stripped == "hello world"


class TestTextBlobTokenization:

    def test_words(self):
        blob = TextBlob("Beautiful is better than ugly. Explicit is better than implicit.")
        assert blob.words == [
            "Beautiful",
            "is",
            "better",
            "than",
            "ugly",
            "Explicit",
            "is",
            "better",
            "than",
            "implicit",
        ]

    def test_words_excludes_punctuation(self):
        blob = TextBlob("Hello, world!")
        assert "," not in blob.words
        assert "!" not in blob.words
        assert "Hello" in blob.words
        assert "world" in blob.words

    def test_tokens_includes_punctuation(self):
        blob = TextBlob("Hello, world!")
        assert "," in blob.tokens
        assert "!" in blob.tokens

    def test_words_includes_apostrophes_in_contractions(self):
        blob = TextBlob("Let's test this.")
        assert "'s" in blob.words
        blob2 = TextBlob("I can't believe it's not butter.")
        assert "n't" in blob2.words
        assert "'s" in blob2.words

    def test_sentences(self):
        text = "Beautiful is better than ugly. Explicit is better than implicit."
        blob = TextBlob(text)
        assert len(blob.sentences) == 2
        assert isinstance(blob.sentences[0], Sentence)
        assert blob.sentences[0].raw == "Beautiful is better than ugly."
        assert blob.sentences[1].raw == "Explicit is better than implicit."

    def test_raw_sentences(self):
        text = "Hello world. How are you?"
        blob = TextBlob(text)
        assert blob.raw_sentences == ["Hello world.", "How are you?"]

    def test_tokenize_method(self):
        blob = TextBlob("Hello world")
        assert blob.tokenize() == ["Hello", "world"]

    def test_custom_tokenizer(self):
        from textblob.base import BaseTokenizer

        class SpaceTokenizer(BaseTokenizer):
            def tokenize(self, text):
                return text.split(" ")

        blob = TextBlob("Hello world test", tokenizer=SpaceTokenizer())
        assert blob.tokens == ["Hello", "world", "test"]


class TestTextBlobStringBehavior:

    def setup_method(self):
        self.text = "Beautiful is better than ugly."
        self.blob = TextBlob(self.text)

    def test_len(self):
        assert len(self.blob) == len(self.text)

    def test_str(self):
        assert str(self.blob) == self.text

    def test_repr(self):
        assert repr(self.blob) == f'TextBlob("{self.text}")'

    def test_iteration(self):
        for i, letter in enumerate(self.blob):
            assert letter == self.text[i]

    def test_in_operator(self):
        assert "better" in self.blob
        assert "worse" not in self.blob

    def test_indexing(self):
        assert self.blob[0] == "B"
        assert self.blob[10] == "i"  # "Beautiful is..." - index 10 is 'i' in 'is'

    def test_slicing(self):
        sliced = self.blob[0:9]
        assert isinstance(sliced, TextBlob)
        assert sliced.raw == "Beautiful"

    def test_upper(self):
        result = self.blob.upper()
        assert isinstance(result, TextBlob)
        assert result == TextBlob(self.text.upper())

    def test_lower(self):
        result = self.blob.lower()
        assert isinstance(result, TextBlob)
        assert result == TextBlob(self.text.lower())

    def test_strip(self):
        blob = TextBlob("  hello  ")
        stripped = blob.strip()
        assert isinstance(stripped, TextBlob)
        assert stripped == TextBlob("hello")

    def test_title(self):
        blob = TextBlob("hello world")
        result = blob.title()
        assert isinstance(result, TextBlob)
        assert result == TextBlob("Hello World")

    def test_find(self):
        assert self.blob.find("better") == self.text.find("better")
        assert self.blob.find("worse") == -1

    def test_rfind(self):
        blob = TextBlob("hello hello hello")
        assert blob.rfind("hello") == 12

    def test_startswith(self):
        assert self.blob.startswith("Beautiful")
        assert self.blob.starts_with("Beautiful")
        assert not self.blob.startswith("ugly")

    def test_endswith(self):
        assert self.blob.endswith("ugly.")
        assert self.blob.ends_with("ugly.")
        assert not self.blob.endswith("beautiful")

    def test_replace(self):
        result = self.blob.replace("ugly", "nice")
        assert isinstance(result, TextBlob)
        assert "nice" in result.raw
        assert "ugly" not in result.raw

    def test_join(self):
        words = ["hello", "world"]
        result = TextBlob(" ").join(words)
        assert isinstance(result, TextBlob)
        assert result == TextBlob("hello world")

    def test_format(self):
        blob = TextBlob("Hello {0}!")
        result = blob.format("world")
        assert isinstance(result, TextBlob)
        assert result == TextBlob("Hello world!")

    def test_split(self):
        result = self.blob.split()
        assert isinstance(result, list)
        assert result == ["Beautiful", "is", "better", "than", "ugly."]

    def test_index(self):
        assert self.blob.index("is") == self.text.index("is")
        with pytest.raises(ValueError):
            self.blob.index("notfound")


class TestTextBlobComparison:

    def test_equality_with_string(self):
        blob = TextBlob("hello")
        assert blob == "hello"
        assert not (blob == "world")

    def test_equality_with_blob(self):
        blob1 = TextBlob("hello")
        blob2 = TextBlob("hello")
        blob3 = TextBlob("world")
        assert blob1 == blob2
        assert not (blob1 == blob3)

    def test_comparison_operators(self):
        blob1 = TextBlob("apple")
        blob2 = TextBlob("banana")
        assert blob1 < blob2
        assert blob1 <= blob2
        assert blob2 > blob1
        assert blob2 >= blob1
        assert blob1 != blob2

    def test_comparison_with_string(self):
        blob = TextBlob("apple")
        assert blob < "banana"
        assert blob > "aardvark"

    def test_invalid_comparison(self):
        blob = TextBlob("one")
        with pytest.raises(TypeError):
            blob < 2  # noqa: B015

    def test_hash(self):
        blob = TextBlob("hello")
        assert hash(blob) == hash("hello")


class TestTextBlobConcatenation:

    def test_add_two_blobs(self):
        blob1 = TextBlob("Hello ")
        blob2 = TextBlob("world!")
        result = blob1 + blob2
        assert isinstance(result, TextBlob)
        assert result == TextBlob("Hello world!")

    def test_add_blob_and_string(self):
        blob = TextBlob("Hello ")
        result = blob + "world!"
        assert isinstance(result, TextBlob)
        assert result == TextBlob("Hello world!")

    def test_add_invalid_type(self):
        blob = TextBlob("Hello")
        with pytest.raises(TypeError):
            blob + 123


class TestTextBlobNgrams:

    def test_ngrams_default(self):
        blob = TextBlob("I am eating a pizza.")
        three_grams = blob.ngrams()
        assert three_grams == [
            ("I", "am", "eating"),
            ("am", "eating", "a"),
            ("eating", "a", "pizza"),
        ]

    def test_ngrams_custom_n(self):
        blob = TextBlob("I am eating a pizza.")
        four_grams = blob.ngrams(n=4)
        assert four_grams == [
            ("I", "am", "eating", "a"),
            ("am", "eating", "a", "pizza"),
        ]

    def test_ngrams_two_grams(self):
        blob = TextBlob("One two three")
        two_grams = blob.ngrams(n=2)
        assert two_grams == [
            ("One", "two"),
            ("two", "three"),
        ]

    def test_ngrams_zero(self):
        blob = TextBlob("Hello world")
        assert blob.ngrams(n=0) == []

    def test_ngrams_negative(self):
        blob = TextBlob("Hello world")
        assert blob.ngrams(n=-1) == []


class TestSentence:

    def test_sentence_creation(self):
        sent = Sentence("Hello world.")
        assert sent.raw == "Hello world."

    def test_sentence_indices(self):
        sent = Sentence("Hello world.", start_index=5, end_index=17)
        assert sent.start == 5
        assert sent.start_index == 5
        assert sent.end == 17
        assert sent.end_index == 17

    def test_sentence_default_indices(self):
        sent = Sentence("Hello.")
        assert sent.start == 0
        assert sent.end == 6

    def test_sentence_inherits_from_textblob(self):
        sent = Sentence("Hello world.")
        assert isinstance(sent, TextBlob)
        assert sent.words == ["Hello", "world"]

    def test_sentence_repr(self):
        sent = Sentence("Hello world.")
        assert repr(sent) == 'Sentence("Hello world.")'


class TestTextBlobWithMultipleSentences:

    def test_sentence_indices_in_blob(self):
        text = "Hello world. How are you?"
        blob = TextBlob(text)
        sent1, sent2 = blob.sentences

        # First sentence
        assert sent1.start == 0
        assert sent1.end == 12
        assert blob.raw[sent1.start:sent1.end] == "Hello world."

        # Second sentence
        assert sent2.start == 13
        assert sent2.end == 25
        assert blob.raw[sent2.start:sent2.end] == "How are you?"

    def test_using_indices_for_slicing(self):
        blob = TextBlob("Hello world. How do you do?")
        sent1, sent2 = blob.sentences
        assert blob[sent1.start:sent1.end] == TextBlob(str(sent1))
        assert blob[sent2.start:sent2.end] == TextBlob(str(sent2))


class TestTextBlobPOSTags:

    def test_pos_tags_returns_list_of_tuples(self):
        """Test that pos_tags returns a list of (word, tag) tuples."""
        blob = TextBlob("Simple is better than complex.")
        tags = blob.pos_tags
        assert isinstance(tags, list)
        assert all(isinstance(item, tuple) for item in tags)
        assert all(len(item) == 2 for item in tags)

    def test_pos_tags_content(self):
        """Test that pos_tags contains expected words."""
        blob = TextBlob("Simple is better than complex.")
        tags = blob.pos_tags
        words = [word for word, tag in tags]
        assert "Simple" in words
        assert "is" in words
        assert "better" in words

    def test_tags_is_alias_for_pos_tags(self):
        """Test that tags is an alias for pos_tags."""
        blob = TextBlob("Hello world.")
        assert blob.tags == blob.pos_tags

    def test_pos_tags_uses_penn_treebank_tagset(self):
        """Test that tags use Penn Treebank tagset."""
        blob = TextBlob("The quick brown fox jumps.")
        tags = blob.pos_tags
        tag_set = {tag for _, tag in tags}
        # Common Penn Treebank tags should appear
        assert any(tag.startswith("DT") for tag in tag_set) or any(
            tag.startswith("NN") for tag in tag_set
        )

    def test_pos_tags_excludes_punctuation(self):
        """Test that punctuation is excluded from pos_tags."""
        blob = TextBlob("Hello, world!")
        tags = blob.pos_tags
        words = [word for word, tag in tags]
        # Punctuation should not be in words
        assert "," not in words
        assert "!" not in words

    def test_sentence_has_pos_tags(self):
        """Test that Sentence objects also have pos_tags."""
        sent = Sentence("The quick brown fox.")
        tags = sent.pos_tags
        assert isinstance(tags, list)
        words = [word for word, tag in tags]
        assert "quick" in words

    def test_custom_pos_tagger(self):
        """Test that custom pos_tagger can be passed to constructor."""
        tagger = NLTKTagger()
        blob = TextBlob("Hello world.", pos_tagger=tagger)
        assert blob.pos_tagger is tagger

    def test_invalid_pos_tagger_raises_error(self):
        """Test that invalid pos_tagger raises ValueError."""
        with pytest.raises(ValueError):
            TextBlob("Hello", pos_tagger="invalid")

    def test_pos_tagger_is_shared_among_instances(self):
        """Test that pos_tagger is shared among instances."""
        blob1 = TextBlob("Hello world")
        blob2 = TextBlob("Another text")
        # Default taggers should be the same instance
        assert blob1.pos_tagger is blob2.pos_tagger

    def test_pos_tags_multiple_sentences(self):
        """Test pos_tags with multiple sentences."""
        blob = TextBlob("Hello world. How are you?")
        tags = blob.pos_tags
        words = [word for word, tag in tags]
        # Words from both sentences should be present
        assert "Hello" in words
        assert "How" in words


class TestTextBlobSentiment:

    def test_sentiment_returns_named_tuple(self):
        """Test that sentiment returns a named tuple with polarity and subjectivity."""
        blob = TextBlob("I love this.")
        sentiment = blob.sentiment
        assert hasattr(sentiment, "polarity")
        assert hasattr(sentiment, "subjectivity")

    def test_polarity_property(self):
        """Test that polarity property returns the polarity score."""
        blob = TextBlob("I love this.")
        assert blob.polarity == blob.sentiment.polarity

    def test_subjectivity_property(self):
        """Test that subjectivity property returns the subjectivity score."""
        blob = TextBlob("I love this.")
        assert blob.subjectivity == blob.sentiment.subjectivity

    def test_positive_sentiment(self):
        """Test that positive text has positive polarity."""
        blob = TextBlob("I love this movie. It's wonderful!")
        assert blob.polarity > 0

    def test_negative_sentiment(self):
        """Test that negative text has negative polarity."""
        blob = TextBlob("I hate this. It's terrible!")
        assert blob.polarity < 0

    def test_polarity_range(self):
        """Test that polarity is within [-1, 1]."""
        blob = TextBlob("This is amazing!")
        assert -1.0 <= blob.polarity <= 1.0

    def test_subjectivity_range(self):
        """Test that subjectivity is within [0, 1]."""
        blob = TextBlob("This is amazing!")
        assert 0.0 <= blob.subjectivity <= 1.0

    def test_sentence_has_sentiment(self):
        """Test that Sentence objects have sentiment properties."""
        sent = Sentence("I love this.")
        assert hasattr(sent, "sentiment")
        assert hasattr(sent, "polarity")
        assert hasattr(sent, "subjectivity")

    def test_custom_analyzer(self):
        """Test that custom analyzer can be passed to constructor."""
        from textblob.sentiments import PatternAnalyzer

        analyzer = PatternAnalyzer()
        blob = TextBlob("Hello world.", analyzer=analyzer)
        assert blob.analyzer is analyzer

    def test_invalid_analyzer_raises_error(self):
        """Test that invalid analyzer raises ValueError."""
        with pytest.raises(ValueError):
            TextBlob("Hello", analyzer="invalid")


class TestJSONSerialization:

    def test_sentence_dict(self):
        """Test Sentence.dict property."""
        sent = Sentence("Hello world.", start_index=0, end_index=12)
        d = sent.dict
        assert isinstance(d, dict)
        assert d["raw"] == "Hello world."
        assert d["start"] == 0
        assert d["end"] == 12
        assert "polarity" in d
        assert "subjectivity" in d

    def test_sentence_json(self):
        """Test Sentence.json method."""
        import json
        sent = Sentence("Hello world.")
        json_str = sent.json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["raw"] == "Hello world."

    def test_sentence_json_with_indent(self):
        """Test Sentence.json with indent parameter."""
        sent = Sentence("Hello world.")
        json_str = sent.json(indent=2)
        assert "\n" in json_str

    def test_textblob_serialized(self):
        """Test TextBlob.serialized property."""
        blob = TextBlob("Hello world. How are you?")
        d = blob.serialized
        assert isinstance(d, dict)
        assert d["raw"] == "Hello world. How are you?"
        assert "sentences" in d
        assert len(d["sentences"]) == 2
        assert "polarity" in d
        assert "subjectivity" in d

    def test_textblob_to_json(self):
        """Test TextBlob.to_json method."""
        import json
        blob = TextBlob("Hello world.")
        json_str = blob.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["raw"] == "Hello world."
        assert "sentences" in data

    def test_textblob_to_json_with_indent(self):
        """Test TextBlob.to_json with indent parameter."""
        blob = TextBlob("Hello world.")
        json_str = blob.to_json(indent=2)
        assert "\n" in json_str

    def test_serialized_sentences_have_indices(self):
        """Test that serialized sentences include start/end indices."""
        blob = TextBlob("Hello world. How are you?")
        d = blob.serialized
        sent1 = d["sentences"][0]
        sent2 = d["sentences"][1]
        assert sent1["start"] == 0
        assert sent1["end"] == 12
        assert sent2["start"] == 13
        assert sent2["end"] == 25


if __name__ == "__main__":
    pytest.main([__file__])
