"""Word inflection functions (pluralize/singularize).

Implements basic pluralization and singularization rules for English.
"""

import re


# Common irregular plural forms
IRREGULAR_PLURALS = {
    "child": "children",
    "man": "men",
    "woman": "women",
    "person": "people",
    "tooth": "teeth",
    "foot": "feet",
    "goose": "geese",
    "mouse": "mice",
    "louse": "lice",
    "ox": "oxen",
    "criterion": "criteria",
    "phenomenon": "phenomena",
    "datum": "data",
    "medium": "media",
    "analysis": "analyses",
    "thesis": "theses",
    "crisis": "crises",
    "hypothesis": "hypotheses",
    "diagnosis": "diagnoses",
    "cactus": "cacti",
    "focus": "foci",
    "fungus": "fungi",
    "nucleus": "nuclei",
    "syllabus": "syllabi",
    "appendix": "appendices",
    "index": "indices",
    "matrix": "matrices",
    "vertex": "vertices",
    "leaf": "leaves",
    "wolf": "wolves",
    "knife": "knives",
    "wife": "wives",
    "life": "lives",
    "self": "selves",
    "elf": "elves",
    "loaf": "loaves",
    "calf": "calves",
    "half": "halves",
}

# Reverse mapping for singularization
IRREGULAR_SINGULARS = {v: k for k, v in IRREGULAR_PLURALS.items()}

# Words that don't change in plural
UNINFLECTED = {
    "sheep", "fish", "deer", "moose", "swine", "bison",
    "salmon", "trout", "aircraft", "spacecraft", "series",
    "species", "corps", "means", "news", "scissors", "pants",
    "shorts", "jeans", "glasses", "headquarters", "premises",
    "cattle", "offspring", "shrimp", "equipment", "information",
}


def pluralize(word, pos="NN"):
    """Return the plural form of a word.

    :param word: The word to pluralize.
    :param pos: Part of speech (default noun).
    :returns: Plural form of the word.
    """
    word_lower = word.lower()

    # Check uninflected words
    if word_lower in UNINFLECTED:
        return word

    # Check irregular plurals
    if word_lower in IRREGULAR_PLURALS:
        plural = IRREGULAR_PLURALS[word_lower]
        # Preserve original capitalization
        if word[0].isupper():
            return plural.capitalize()
        return plural

    # Apply regular rules
    # Words ending in 's', 'x', 'z', 'ch', 'sh' add 'es'
    if re.search(r"(s|x|z|ch|sh)$", word_lower):
        return word + "es"

    # Words ending in consonant + 'y' change 'y' to 'ies'
    if re.search(r"[^aeiou]y$", word_lower):
        return word[:-1] + "ies"

    # Words ending in 'f' or 'fe' change to 'ves' (except some)
    if re.search(r"(f|fe)$", word_lower) and word_lower not in {
        "roof", "proof", "belief", "chief", "chef", "cliff", "safe"
    }:
        if word.endswith("fe"):
            return word[:-2] + "ves"
        elif word.endswith("f"):
            return word[:-1] + "ves"

    # Words ending in consonant + 'o' add 'es'
    if re.search(r"[^aeiou]o$", word_lower) and word_lower not in {
        "photo", "piano", "memo", "auto", "zero", "pro", "logo"
    }:
        return word + "es"

    # Default: add 's'
    return word + "s"


def singularize(word, pos="NN"):
    """Return the singular form of a word.

    :param word: The word to singularize.
    :param pos: Part of speech (default noun).
    :returns: Singular form of the word.
    """
    word_lower = word.lower()

    # Check uninflected words
    if word_lower in UNINFLECTED:
        return word

    # Check irregular singulars
    if word_lower in IRREGULAR_SINGULARS:
        singular = IRREGULAR_SINGULARS[word_lower]
        # Preserve original capitalization
        if word[0].isupper():
            return singular.capitalize()
        return singular

    # Apply regular rules
    # Words ending in 'ies' -> 'y' (if preceded by consonant)
    if re.search(r"[^aeiou]ies$", word_lower):
        return word[:-3] + "y"

    # Words ending in 'ves' -> 'f' or 'fe'
    if word_lower.endswith("ves"):
        # Check if it should be 'fe' instead of 'f'
        stem = word[:-3]
        if stem.lower() in {"wi", "li", "kni"}:  # wife, life, knife
            return stem + "fe"
        return stem + "f"

    # Words ending in 'es' after 's', 'x', 'z', 'ch', 'sh'
    if re.search(r"(ss|x|z|ch|sh)es$", word_lower):
        return word[:-2]

    # Words ending in 'oes' (where singular is 'o')
    if word_lower.endswith("oes") and word_lower not in {"shoes", "does", "goes"}:
        return word[:-2]

    # Words ending in 'ses' after regular 's'
    if word_lower.endswith("ses") and not word_lower.endswith("sses"):
        return word[:-1]

    # Words ending in 's' (not 'ss')
    if word_lower.endswith("s") and not word_lower.endswith("ss"):
        return word[:-1]

    return word
