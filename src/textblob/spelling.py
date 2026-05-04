"""Spelling correction for TextBlob.

Uses NLTK's words corpus for vocabulary and implements a simple
edit-distance based spell checker.
"""

from collections import Counter
import re

import nltk

from textblob.decorators import requires_nltk_corpus


class SpellChecker:
    """A simple spell checker based on edit distance.

    Uses the NLTK words corpus as a vocabulary.
    """

    def __init__(self):
        self._words = None
        self._word_counts = None

    @property
    @requires_nltk_corpus
    def words(self):
        """Return the word vocabulary set."""
        if self._words is None:
            from nltk.corpus import words
            self._words = set(w.lower() for w in words.words())
        return self._words

    @property
    @requires_nltk_corpus
    def word_counts(self):
        """Return word frequency counts from brown corpus."""
        if self._word_counts is None:
            from nltk.corpus import brown
            self._word_counts = Counter(w.lower() for w in brown.words())
        return self._word_counts

    def _edits1(self, word):
        """Return all edits that are one edit away from word."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        """Return all edits that are two edits away from word."""
        return set(e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))

    def _known(self, words):
        """Return the subset of words that are in the vocabulary."""
        return set(w for w in words if w in self.words)

    def candidates(self, word):
        """Generate possible spelling corrections for word.

        :returns: Set of candidate words.
        """
        word_lower = word.lower()
        # First check if word is correct
        if word_lower in self.words:
            return {word_lower}
        # Then check edits of distance 1
        candidates = self._known(self._edits1(word_lower))
        if candidates:
            return candidates
        # Then check edits of distance 2
        candidates = self._known(self._edits2(word_lower))
        if candidates:
            return candidates
        # Return original word if no candidates found
        return {word_lower}

    def suggest(self, word):
        """Return list of (suggestion, confidence) tuples for a word.

        :param word: The word to check.
        :returns: List of (suggested_word, confidence) tuples sorted by confidence.
        """
        word_lower = word.lower()
        candidates_set = self.candidates(word_lower)

        # Get word frequencies
        total = sum(self.word_counts.get(c, 1) for c in candidates_set)
        suggestions = []
        for candidate in candidates_set:
            freq = self.word_counts.get(candidate, 1)
            confidence = freq / total if total > 0 else 0.0
            suggestions.append((candidate, confidence))

        # Sort by confidence descending
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions

    def correct(self, word):
        """Return the most likely correct spelling of a word.

        :param word: The word to correct.
        :returns: The corrected word string.
        """
        word_lower = word.lower()
        suggestions = self.suggest(word_lower)
        if suggestions:
            corrected = suggestions[0][0]
            # Preserve original capitalization
            if word[0].isupper():
                return corrected.capitalize()
            return corrected
        return word


# Singleton instance
_spell_checker = None


def get_spell_checker():
    """Get the singleton spell checker instance."""
    global _spell_checker
    if _spell_checker is None:
        _spell_checker = SpellChecker()
    return _spell_checker


def suggest(word):
    """Return list of (suggestion, confidence) tuples for a word.

    :param word: The word to check.
    :returns: List of (suggested_word, confidence) tuples.
    """
    return get_spell_checker().suggest(word)


def correct(word):
    """Return the most likely correct spelling of a word.

    :param word: The word to correct.
    :returns: The corrected word string.
    """
    return get_spell_checker().correct(word)
