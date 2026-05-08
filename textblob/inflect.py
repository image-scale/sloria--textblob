"""Word inflection functions for TextBlob.

This module provides functions for pluralizing and singularizing English words.
"""

IRREGULAR_PLURALS = {
    'child': 'children',
    'man': 'men',
    'woman': 'women',
    'tooth': 'teeth',
    'foot': 'feet',
    'goose': 'geese',
    'mouse': 'mice',
    'louse': 'lice',
    'person': 'people',
    'ox': 'oxen',
    'criterion': 'criteria',
    'phenomenon': 'phenomena',
    'analysis': 'analyses',
    'basis': 'bases',
    'crisis': 'crises',
    'diagnosis': 'diagnoses',
    'hypothesis': 'hypotheses',
    'thesis': 'theses',
    'appendix': 'appendices',
    'index': 'indices',
    'matrix': 'matrices',
    'vertex': 'vertices',
    'focus': 'foci',
    'radius': 'radii',
    'stimulus': 'stimuli',
    'cactus': 'cacti',
    'fungus': 'fungi',
    'nucleus': 'nuclei',
    'syllabus': 'syllabi',
    'alumnus': 'alumni',
    'bacterium': 'bacteria',
    'curriculum': 'curricula',
    'datum': 'data',
    'medium': 'media',
    'memorandum': 'memoranda',
    'stratum': 'strata',
    'leaf': 'leaves',
    'knife': 'knives',
    'life': 'lives',
    'wife': 'wives',
    'half': 'halves',
    'calf': 'calves',
    'loaf': 'loaves',
    'self': 'selves',
    'shelf': 'shelves',
    'wolf': 'wolves',
    'elf': 'elves',
    'thief': 'thieves',
    'scarf': 'scarves',
    'die': 'dice',
    'penny': 'pence',
    'fish': 'fish',
    'sheep': 'sheep',
    'deer': 'deer',
    'species': 'species',
    'series': 'series',
    'aircraft': 'aircraft',
    'moose': 'moose',
    'swine': 'swine',
    'bison': 'bison',
    'salmon': 'salmon',
    'trout': 'trout',
}

IRREGULAR_SINGULARS = {v: k for k, v in IRREGULAR_PLURALS.items()}

UNCOUNTABLE = {
    'equipment', 'information', 'rice', 'money', 'species', 'series',
    'fish', 'sheep', 'deer', 'aircraft', 'moose', 'swine', 'bison',
    'salmon', 'trout', 'news', 'mathematics', 'physics', 'economics',
    'ethics', 'politics', 'athletics', 'electronics', 'mechanics',
    'advice', 'bread', 'butter', 'cheese', 'coffee', 'furniture',
    'garbage', 'homework', 'knowledge', 'luggage', 'music', 'sugar',
    'tea', 'water', 'weather', 'work', 'traffic', 'research',
}


def pluralize(word, pos='NOUN', custom=None, classical=True):
    """Return the plural form of a word.

    Args:
        word: The word to pluralize
        pos: Part of speech (NOUN, VERB, ADJ)
        custom: Optional dict of custom plural forms
        classical: If True, use classical/Latinate plurals

    Returns:
        The plural form of the word.
    """
    if custom and word.lower() in custom:
        return custom[word.lower()]

    lower = word.lower()

    if lower in UNCOUNTABLE:
        return word

    if lower in IRREGULAR_PLURALS:
        plural = IRREGULAR_PLURALS[lower]
        if word[0].isupper():
            return plural.capitalize()
        return plural

    if lower.endswith('man') and len(lower) > 3:
        base = word[:-3]
        ending = word[-3:]
        if ending == 'Man':
            return base + 'Men'
        elif ending == 'MAN':
            return base + 'MEN'
        else:
            return base + 'men'

    if '-' in word:
        parts = word.split('-')
        parts[0] = pluralize(parts[0], pos, custom, classical)
        return '-'.join(parts)

    if lower.endswith("'s"):
        return pluralize(word[:-2], pos, custom, classical) + "s'"
    if lower.endswith("s'"):
        return word

    if lower.endswith('sis'):
        return word[:-2] + 'es'
    if lower.endswith('xis'):
        return word[:-2] + 'es'

    if classical:
        if lower.endswith('us') and len(lower) > 2:
            if lower.endswith('ous'):
                pass
            elif lower in ('focus', 'radius', 'stimulus', 'cactus', 'fungus',
                           'nucleus', 'syllabus', 'alumnus'):
                return word[:-2] + 'i'

        if lower.endswith('um') and len(lower) > 2:
            if lower in ('bacterium', 'curriculum', 'datum', 'medium',
                         'memorandum', 'stratum', 'stadium'):
                return word[:-2] + 'a'

        if lower.endswith('on') and len(lower) > 2:
            if lower in ('criterion', 'phenomenon'):
                return word[:-2] + 'a'

    if lower.endswith('y'):
        if len(lower) > 1 and lower[-2] in 'aeiou':
            return word + 's'
        else:
            return word[:-1] + 'ies'

    if lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
        return word + 'es'

    if lower.endswith('fe'):
        return word[:-2] + 'ves'

    if lower.endswith('f'):
        if lower in ('roof', 'chief', 'belief', 'proof', 'brief', 'cliff',
                     'gulf', 'grief', 'staff', 'reef', 'safe', 'dwarf'):
            return word + 's'
        return word[:-1] + 'ves'

    if lower.endswith('o'):
        if lower in ('photo', 'piano', 'video', 'radio', 'memo', 'zoo',
                     'logo', 'studio', 'pro', 'auto', 'euro'):
            return word + 's'
        if len(lower) > 1 and lower[-2] in 'aeiou':
            return word + 's'
        return word + 'es'

    return word + 's'


def singularize(word, pos='NOUN', custom=None):
    """Return the singular form of a word.

    Args:
        word: The word to singularize
        pos: Part of speech (NOUN, VERB, ADJ)
        custom: Optional dict of custom singular forms

    Returns:
        The singular form of the word.
    """
    if custom and word.lower() in custom:
        return custom[word.lower()]

    lower = word.lower()

    if lower in UNCOUNTABLE:
        return word

    if lower in IRREGULAR_SINGULARS:
        singular = IRREGULAR_SINGULARS[lower]
        if word[0].isupper():
            return singular.capitalize()
        return singular

    if lower.endswith('men') and len(lower) > 3:
        base = word[:-3]
        ending = word[-3:]
        if ending == 'Men':
            return base + 'Man'
        elif ending == 'MEN':
            return base + 'MAN'
        else:
            return base + 'man'

    if '-' in word:
        parts = word.split('-')
        parts[0] = singularize(parts[0], pos, custom)
        return '-'.join(parts)

    if lower.endswith("s'"):
        return singularize(word[:-2], pos, custom) + "'s"

    if lower.endswith('ses') and len(lower) > 3:
        if lower.endswith('yses'):
            return word[:-3] + 'sis'
        if lower.endswith('ases') or lower.endswith('eses') or lower.endswith('ises'):
            return word[:-2]
        return word[:-2]

    if lower.endswith('xes'):
        if lower.endswith('ixes') and len(lower) > 4:
            return word[:-4] + 'ix'
        if lower.endswith('ices') and len(lower) > 4:
            return word[:-4] + 'ex'
        return word[:-2]

    if lower.endswith('i') and len(lower) > 1:
        if lower in ('alumni', 'cacti', 'foci', 'fungi', 'nuclei',
                     'radii', 'stimuli', 'syllabi'):
            return word[:-1] + 'us'

    if lower.endswith('a') and len(lower) > 1:
        if lower in ('bacteria', 'curricula', 'data', 'media',
                     'memoranda', 'strata', 'criteria', 'phenomena'):
            if lower.endswith('ria') or lower.endswith('ena'):
                return word[:-1] + 'on'
            return word[:-1] + 'um'

    if lower.endswith('ves'):
        if lower.endswith('lves') and len(lower) > 4:
            base = word[:-3]
            if lower[:-3] in ('ca', 'ha', 'she', 'wo', 'li'):
                return base + 'f'
            return base + 'fe'
        if lower.endswith('ives'):
            if lower in ('wives', 'knives', 'lives'):
                return word[:-3] + 'fe'
            return word[:-1]
        if lower.endswith('aves') or lower.endswith('eaves'):
            return word[:-1]
        return word[:-3] + 'f'

    if lower.endswith('ies') and len(lower) > 3:
        if lower[-4] in 'aeiou':
            return word[:-1]
        return word[:-3] + 'y'

    if lower.endswith('oes'):
        if lower in ('potatoes', 'tomatoes', 'heroes', 'echoes', 'torpedoes',
                     'vetoes', 'embargoes', 'cargoes', 'mosquitoes', 'volcanoes'):
            return word[:-2]
        return word[:-1]

    if lower.endswith('ches') or lower.endswith('shes'):
        return word[:-2]

    if lower.endswith('sses') or lower.endswith('zzes'):
        return word[:-2]

    if lower.endswith('es') and len(lower) > 2:
        if lower[-3] in 'sxz':
            return word[:-2]
        return word[:-1]

    if lower.endswith('s') and len(lower) > 1:
        if lower.endswith('ss'):
            return word
        if lower.endswith('us') or lower.endswith('is'):
            return word
        return word[:-1]

    return word
