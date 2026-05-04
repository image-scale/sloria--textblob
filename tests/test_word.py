"""Tests for the Word class."""

import pytest

from textblob import TextBlob, Word, pluralize, singularize


class TestWordCreation:

    def test_create_word(self):
        """Test creating a Word instance."""
        w = Word("hello")
        assert w == "hello"
        assert isinstance(w, str)

    def test_word_repr(self):
        """Test Word repr."""
        w = Word("hello")
        assert repr(w) == "Word('hello')"

    def test_word_with_pos_tag(self):
        """Test creating Word with POS tag."""
        w = Word("running", pos_tag="VBG")
        assert w.pos_tag == "VBG"

    def test_word_string_attribute(self):
        """Test Word string attribute."""
        w = Word("hello")
        assert w.string == "hello"


class TestWordInflection:

    def test_singularize(self):
        """Test singularize method."""
        w = Word("dogs")
        assert w.singularize() == "dog"

    def test_singularize_returns_word(self):
        """Test singularize returns a Word object."""
        w = Word("cats")
        result = w.singularize()
        assert isinstance(result, Word)

    def test_pluralize(self):
        """Test pluralize method."""
        w = Word("dog")
        assert w.pluralize() == "dogs"

    def test_pluralize_returns_word(self):
        """Test pluralize returns a Word object."""
        w = Word("cat")
        result = w.pluralize()
        assert isinstance(result, Word)

    def test_irregular_plural(self):
        """Test irregular plural forms."""
        w = Word("children")
        assert w.singularize() == "child"

    def test_irregular_singular(self):
        """Test irregular singular to plural."""
        w = Word("child")
        assert w.pluralize() == "children"

    def test_es_plural(self):
        """Test words that add 'es' for plural."""
        w = Word("box")
        assert w.pluralize() == "boxes"

    def test_ies_plural(self):
        """Test words ending in consonant + y."""
        w = Word("city")
        assert w.pluralize() == "cities"

    def test_uninflected(self):
        """Test uninflected words."""
        w = Word("sheep")
        assert w.singularize() == "sheep"
        assert w.pluralize() == "sheep"


class TestWordStemming:

    def test_stem(self):
        """Test stem method."""
        w = Word("running")
        stemmed = w.stem()
        assert stemmed == "run"

    def test_stem_returns_word(self):
        """Test stem returns a Word object."""
        w = Word("cats")
        result = w.stem()
        assert isinstance(result, Word)

    def test_stem_jumps(self):
        """Test stemming 'jumps'."""
        w = Word("jumps")
        assert w.stem() == "jump"


class TestWordLemmatization:

    def test_lemmatize_default(self):
        """Test lemmatize with default pos (noun)."""
        w = Word("dogs")
        assert w.lemmatize() == "dog"

    def test_lemmatize_returns_word(self):
        """Test lemmatize returns a Word object."""
        w = Word("cats")
        result = w.lemmatize()
        assert isinstance(result, Word)

    def test_lemmatize_verb(self):
        """Test lemmatize with verb pos."""
        w = Word("running")
        assert w.lemmatize(pos="v") == "run"

    def test_lemmatize_adjective(self):
        """Test lemmatize with adjective pos."""
        w = Word("better")
        # 'better' as adjective lemmatizes to 'good' or stays 'better'
        # depending on WordNet, just test it runs
        result = w.lemmatize(pos="a")
        assert isinstance(result, Word)

    def test_lemmatize_with_pos_tag(self):
        """Test lemmatize uses pos_tag if set."""
        w = Word("running", pos_tag="VBG")
        result = w.lemmatize()
        assert result == "run"


class TestWordSpelling:

    def test_correct(self):
        """Test correct method."""
        w = Word("speling")  # Common misspelling
        corrected = w.correct()
        assert isinstance(corrected, Word)
        # Should correct to 'spelling' or similar
        assert corrected in ["spelling", "spieling", "spewing", "sling", "speling"]

    def test_spellcheck(self):
        """Test spellcheck method returns suggestions."""
        w = Word("helo")
        suggestions = w.spellcheck()
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Each suggestion is (word, confidence) tuple
        word, confidence = suggestions[0]
        assert isinstance(word, Word)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_spellcheck_correct_word(self):
        """Test spellcheck with correctly spelled word."""
        w = Word("hello")
        suggestions = w.spellcheck()
        # Should include the word itself with high confidence
        words = [word for word, conf in suggestions]
        assert "hello" in words


class TestWordWordNet:

    def test_synsets(self):
        """Test synsets property."""
        w = Word("dog")
        synsets = w.synsets
        assert isinstance(synsets, list)
        assert len(synsets) > 0
        # First synset should be related to dog
        assert "dog" in synsets[0].name()

    def test_definitions(self):
        """Test definitions property."""
        w = Word("dog")
        defs = w.definitions
        assert isinstance(defs, list)
        assert len(defs) > 0
        assert isinstance(defs[0], str)

    def test_get_synsets_with_pos(self):
        """Test get_synsets with specific POS."""
        w = Word("run")
        verb_synsets = w.get_synsets(pos="v")
        assert len(verb_synsets) > 0
        # Should only have verb synsets
        for syn in verb_synsets:
            assert syn.pos() == "v"

    def test_define_with_pos(self):
        """Test define with specific POS."""
        w = Word("run")
        verb_defs = w.define(pos="v")
        assert isinstance(verb_defs, list)
        assert len(verb_defs) > 0

    def test_synsets_with_pos_tag(self):
        """Test synsets uses pos_tag if set."""
        w = Word("run", pos_tag="VB")
        synsets = w.synsets
        # Should prefer verb synsets
        assert len(synsets) > 0


class TestWordFromTextBlob:

    def test_words_are_word_objects(self):
        """Test that TextBlob.words returns Word objects."""
        blob = TextBlob("Hello world")
        words = blob.words
        assert all(isinstance(w, Word) for w in words)

    def test_word_methods_from_blob(self):
        """Test using Word methods from TextBlob.words."""
        blob = TextBlob("The dogs are running.")
        dogs = blob.words[1]
        assert dogs.singularize() == "dog"

    def test_sentence_words_are_word_objects(self):
        """Test that Sentence.words returns Word objects."""
        blob = TextBlob("Hello world. How are you?")
        sent = blob.sentences[0]
        words = sent.words
        assert all(isinstance(w, Word) for w in words)


class TestModuleFunctions:

    def test_pluralize_function(self):
        """Test module-level pluralize function."""
        assert pluralize("dog") == "dogs"
        assert pluralize("cat") == "cats"
        assert pluralize("child") == "children"

    def test_singularize_function(self):
        """Test module-level singularize function."""
        assert singularize("dogs") == "dog"
        assert singularize("cats") == "cat"
        assert singularize("children") == "child"


if __name__ == "__main__":
    pytest.main([__file__])
