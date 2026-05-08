"""Tests for TextBlob."""

import pytest
from textblob import TextBlob, Word, WordList, Sentence, Blobber


class TestWord:
    """Tests for the Word class."""

    def test_word_creation(self):
        word = Word("hello")
        assert str(word) == "hello"
        assert isinstance(word, str)

    def test_singularize(self):
        assert Word("dogs").singularize() == "dog"
        assert Word("children").singularize() == "child"
        assert Word("mice").singularize() == "mouse"

    def test_pluralize(self):
        assert Word("dog").pluralize() == "dogs"
        assert Word("child").pluralize() == "children"
        assert Word("mouse").pluralize() == "mice"

    def test_spellcheck(self):
        word = Word("helo")
        suggestions = word.spellcheck()
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_correct(self):
        word = Word("speling")
        corrected = word.correct()
        assert isinstance(corrected, Word)


class TestWordList:
    """Tests for the WordList class."""

    def test_wordlist_creation(self):
        wl = WordList(["hello", "world"])
        assert len(wl) == 2
        assert isinstance(wl[0], Word)

    def test_count(self):
        wl = WordList(["hello", "world", "hello"])
        assert wl.count("hello") == 2
        assert wl.count("HELLO") == 2
        assert wl.count("HELLO", case_sensitive=True) == 0

    def test_upper_lower(self):
        wl = WordList(["Hello", "World"])
        assert wl.upper() == WordList(["HELLO", "WORLD"])
        assert wl.lower() == WordList(["hello", "world"])

    def test_singularize_pluralize(self):
        wl = WordList(["dogs", "cats"])
        assert wl.singularize() == WordList(["dog", "cat"])

        wl = WordList(["dog", "cat"])
        assert wl.pluralize() == WordList(["dogs", "cats"])

    def test_slicing(self):
        wl = WordList(["a", "b", "c", "d"])
        assert isinstance(wl[1:3], WordList)
        assert wl[1:3] == WordList(["b", "c"])


class TestTextBlob:
    """Tests for the TextBlob class."""

    def test_creation(self):
        blob = TextBlob("Hello world.")
        assert blob.raw == "Hello world."
        assert str(blob) == "Hello world."

    def test_words(self):
        blob = TextBlob("Hello world!")
        assert isinstance(blob.words, WordList)
        assert "Hello" in blob.words
        assert "world" in blob.words

    def test_sentences(self):
        blob = TextBlob("Hello world. How are you?")
        assert len(blob.sentences) == 2
        assert isinstance(blob.sentences[0], Sentence)

    def test_sentiment(self):
        blob = TextBlob("This is amazing and wonderful!")
        assert hasattr(blob.sentiment, 'polarity')
        assert hasattr(blob.sentiment, 'subjectivity')
        assert -1 <= blob.sentiment.polarity <= 1
        assert 0 <= blob.sentiment.subjectivity <= 1

    def test_ngrams(self):
        blob = TextBlob("one two three four five")
        bigrams = blob.ngrams(n=2)
        assert len(bigrams) == 4
        assert isinstance(bigrams[0], WordList)

    def test_word_counts(self):
        blob = TextBlob("Hello hello world")
        assert blob.word_counts['hello'] == 2
        assert blob.word_counts['world'] == 1

    def test_string_methods(self):
        blob = TextBlob("Hello World")
        assert blob.upper().raw == "HELLO WORLD"
        assert blob.lower().raw == "hello world"
        assert blob.startswith("Hello")
        assert blob.endswith("World")
        assert len(blob) == 11

    def test_json(self):
        blob = TextBlob("Hello world.")
        json_str = blob.json
        assert isinstance(json_str, str)
        assert "Hello world." in json_str


class TestSentence:
    """Tests for the Sentence class."""

    def test_sentence_indices(self):
        blob = TextBlob("Hello world. How are you?")
        sent = blob.sentences[0]
        assert sent.start_index == 0
        assert sent.end_index > 0

    def test_sentence_dict(self):
        blob = TextBlob("This is great!")
        sent = blob.sentences[0]
        d = sent.dict
        assert 'raw' in d
        assert 'polarity' in d
        assert 'subjectivity' in d


class TestBlobber:
    """Tests for the Blobber class."""

    def test_blobber_factory(self):
        tb = Blobber()
        blob1 = tb("Hello world.")
        blob2 = tb("Goodbye world.")
        assert isinstance(blob1, TextBlob)
        assert isinstance(blob2, TextBlob)


class TestInflection:
    """Tests for word inflection."""

    def test_irregular_plurals(self):
        from textblob.inflect import pluralize, singularize

        assert pluralize("child") == "children"
        assert pluralize("woman") == "women"
        assert pluralize("man") == "men"
        assert pluralize("tooth") == "teeth"
        assert pluralize("foot") == "feet"
        assert pluralize("goose") == "geese"
        assert pluralize("mouse") == "mice"

        assert singularize("children") == "child"
        assert singularize("women") == "woman"
        assert singularize("men") == "man"
        assert singularize("teeth") == "tooth"
        assert singularize("feet") == "foot"
        assert singularize("geese") == "goose"
        assert singularize("mice") == "mouse"

    def test_regular_plurals(self):
        from textblob.inflect import pluralize, singularize

        assert pluralize("dog") == "dogs"
        assert pluralize("cat") == "cats"
        assert pluralize("box") == "boxes"
        assert pluralize("church") == "churches"
        assert pluralize("baby") == "babies"

        assert singularize("dogs") == "dog"
        assert singularize("cats") == "cat"
        assert singularize("boxes") == "box"


class TestTokenizers:
    """Tests for tokenizers."""

    def test_word_tokenize(self):
        from textblob.tokenizers import word_tokenize

        tokens = word_tokenize("Hello, world!")
        assert "Hello" in tokens
        assert "world" in tokens

    def test_sent_tokenize(self):
        from textblob.tokenizers import sent_tokenize

        sentences = list(sent_tokenize("Hello world. How are you?"))
        assert len(sentences) == 2


class TestSpelling:
    """Tests for spelling correction."""

    def test_correction(self):
        from textblob.spelling import correct

        assert correct("speling") == "spelling"
        assert correct("hello") == "hello"

    def test_spellcheck(self):
        from textblob.spelling import spellcheck

        suggestions = spellcheck("helo")
        assert len(suggestions) > 0
        words = [w for w, _ in suggestions]
        assert "hello" in words or "help" in words


class TestSentiment:
    """Tests for sentiment analysis."""

    def test_pattern_analyzer(self):
        from textblob.sentiments import PatternAnalyzer

        analyzer = PatternAnalyzer()
        sentiment = analyzer.analyze("This is great and wonderful!")

        assert sentiment.polarity > 0
        assert 0 <= sentiment.subjectivity <= 1

    def test_negative_sentiment(self):
        from textblob.sentiments import PatternAnalyzer

        analyzer = PatternAnalyzer()
        sentiment = analyzer.analyze("This is terrible and awful!")

        assert sentiment.polarity < 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
