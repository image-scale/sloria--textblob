"""Spelling correction for TextBlob.

Implements Peter Norvig's spelling correction algorithm.
"""

import re
from collections import Counter


class SpellingCorrector:
    """Spelling corrector using Norvig's algorithm."""

    def __init__(self, word_counts=None):
        """Initialize the spelling corrector.

        Args:
            word_counts: Optional dict of word frequencies. If None,
                        uses the default English word frequency dictionary.
        """
        if word_counts is None:
            from .en import WORD_COUNTS
            word_counts = WORD_COUNTS
        self.word_counts = word_counts
        self._total = sum(word_counts.values())

    def probability(self, word):
        """Return the probability of a word based on frequency.

        Args:
            word: The word to check

        Returns:
            Probability of the word (0.0 to 1.0).
        """
        return self.word_counts.get(word.lower(), 0) / self._total

    def correction(self, word):
        """Return the most probable spelling correction for a word.

        Args:
            word: The word to correct

        Returns:
            The most likely correct spelling.
        """
        candidates = self.candidates(word)
        return max(candidates, key=self.probability)

    def candidates(self, word):
        """Generate possible spelling corrections for a word.

        Returns candidates in order of edit distance:
        1. Known words (edit distance 0)
        2. Edit distance 1 candidates
        3. Edit distance 2 candidates
        4. Original word (if no corrections found)

        Args:
            word: The word to generate candidates for

        Returns:
            Set of candidate corrections.
        """
        lower = word.lower()
        return (
            self._known([lower]) or
            self._known(self._edits1(lower)) or
            self._known(self._edits2(lower)) or
            {lower}
        )

    def spellcheck(self, word):
        """Return spelling suggestions with confidence scores.

        Args:
            word: The word to check

        Returns:
            List of (word, confidence) tuples, sorted by confidence.
        """
        candidates = self.candidates(word)
        suggestions = []
        for candidate in candidates:
            prob = self.probability(candidate)
            suggestions.append((candidate, prob))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions

    def _known(self, words):
        """Return the subset of words that are in the dictionary.

        Args:
            words: Iterable of words to check

        Returns:
            Set of known words.
        """
        return set(w for w in words if w in self.word_counts)

    def _edits1(self, word):
        """Generate all edits that are one edit away from word.

        Edit types:
        - Deletes: remove one character
        - Transposes: swap adjacent characters
        - Replaces: change one character to another
        - Inserts: add one character

        Args:
            word: The word to generate edits for

        Returns:
            Set of all edit-distance-1 words.
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        """Generate all edits that are two edits away from word.

        Args:
            word: The word to generate edits for

        Returns:
            Generator of all edit-distance-2 words.
        """
        return (e2 for e1 in self._edits1(word) for e2 in self._edits1(e1))


_corrector = None


def get_corrector():
    """Get the singleton spelling corrector instance."""
    global _corrector
    if _corrector is None:
        _corrector = SpellingCorrector()
    return _corrector


def correct(word):
    """Return the most probable spelling correction for a word.

    Args:
        word: The word to correct

    Returns:
        The most likely correct spelling.
    """
    return get_corrector().correction(word)


def spellcheck(word):
    """Return spelling suggestions with confidence scores.

    Args:
        word: The word to check

    Returns:
        List of (word, confidence) tuples.
    """
    return get_corrector().spellcheck(word)
