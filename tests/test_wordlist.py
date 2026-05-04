"""Tests for the WordList class."""

import pytest

from textblob import TextBlob, Word, WordList


class TestWordListCreation:

    def test_create_wordlist_from_list(self):
        """Test creating a WordList from a list of strings."""
        words = WordList(["hello", "world"])
        assert len(words) == 2
        assert words[0] == "hello"
        assert words[1] == "world"

    def test_wordlist_is_list(self):
        """Test that WordList inherits from list."""
        words = WordList(["hello", "world"])
        assert isinstance(words, list)

    def test_wordlist_contains_word_objects(self):
        """Test that WordList converts strings to Word objects."""
        words = WordList(["hello", "world"])
        assert all(isinstance(w, Word) for w in words)

    def test_wordlist_repr(self):
        """Test WordList repr."""
        words = WordList(["hello", "world"])
        assert "WordList" in repr(words)

    def test_empty_wordlist(self):
        """Test creating an empty WordList."""
        words = WordList()
        assert len(words) == 0

    def test_wordlist_slicing_returns_wordlist(self):
        """Test that slicing a WordList returns a WordList."""
        words = WordList(["one", "two", "three", "four"])
        sliced = words[1:3]
        assert isinstance(sliced, WordList)
        assert sliced == WordList(["two", "three"])


class TestWordListCount:

    def test_count_case_insensitive(self):
        """Test case-insensitive counting."""
        words = WordList(["Hello", "hello", "HELLO", "world"])
        assert words.count("hello") == 3

    def test_count_case_sensitive(self):
        """Test case-sensitive counting."""
        words = WordList(["Hello", "hello", "HELLO", "world"])
        assert words.count("Hello", case_sensitive=True) == 1

    def test_count_not_found(self):
        """Test counting word not in list."""
        words = WordList(["hello", "world"])
        assert words.count("foo") == 0


class TestWordListTransformations:

    def test_upper(self):
        """Test upper method."""
        words = WordList(["hello", "world"])
        result = words.upper()
        assert isinstance(result, WordList)
        assert result == WordList(["HELLO", "WORLD"])

    def test_lower(self):
        """Test lower method."""
        words = WordList(["Hello", "WORLD"])
        result = words.lower()
        assert isinstance(result, WordList)
        assert result == WordList(["hello", "world"])

    def test_singularize(self):
        """Test singularize method."""
        words = WordList(["dogs", "cats", "mice"])
        result = words.singularize()
        assert isinstance(result, WordList)
        assert result == WordList(["dog", "cat", "mouse"])

    def test_pluralize(self):
        """Test pluralize method."""
        words = WordList(["dog", "cat", "mouse"])
        result = words.pluralize()
        assert isinstance(result, WordList)
        assert result == WordList(["dogs", "cats", "mice"])

    def test_lemmatize(self):
        """Test lemmatize method."""
        words = WordList(["running", "dogs", "better"])
        result = words.lemmatize()
        assert isinstance(result, WordList)
        # Default lemmatizes as noun
        assert all(isinstance(w, Word) for w in result)

    def test_lemmatize_with_pos(self):
        """Test lemmatize with specific POS."""
        words = WordList(["running", "jumping", "walking"])
        result = words.lemmatize(pos="v")
        assert isinstance(result, WordList)
        assert result == WordList(["run", "jump", "walk"])

    def test_stem(self):
        """Test stem method."""
        words = WordList(["running", "jumps", "easily"])
        result = words.stem()
        assert isinstance(result, WordList)
        assert result == WordList(["run", "jump", "easili"])


class TestWordListModification:

    def test_append(self):
        """Test append method."""
        words = WordList(["hello"])
        words.append("world")
        assert len(words) == 2
        assert isinstance(words[1], Word)

    def test_extend(self):
        """Test extend method."""
        words = WordList(["hello"])
        words.extend(["beautiful", "world"])
        assert len(words) == 3
        assert all(isinstance(w, Word) for w in words)


class TestWordListFromTextBlob:

    def test_textblob_words_returns_wordlist(self):
        """Test that TextBlob.words returns a WordList."""
        blob = TextBlob("Hello beautiful world")
        words = blob.words
        assert isinstance(words, WordList)

    def test_wordlist_methods_from_blob(self):
        """Test using WordList methods from TextBlob.words."""
        blob = TextBlob("The dogs are running quickly")
        words = blob.words

        # Test singularize
        singular = words.singularize()
        assert isinstance(singular, WordList)

        # Test upper
        upper = words.upper()
        assert all(str(w).isupper() for w in upper)

    def test_wordlist_count_from_blob(self):
        """Test counting words in TextBlob.words."""
        blob = TextBlob("The cat and the Cat and THE CAT")
        words = blob.words
        assert words.count("cat") == 3
        assert words.count("the") == 3

    def test_sentence_words_returns_wordlist(self):
        """Test that Sentence.words also returns a WordList."""
        blob = TextBlob("Hello world. How are you?")
        sent = blob.sentences[0]
        words = sent.words
        assert isinstance(words, WordList)


if __name__ == "__main__":
    pytest.main([__file__])
