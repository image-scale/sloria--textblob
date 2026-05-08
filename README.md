# TextBlob Clone

A Python library for processing textual data. It provides a simple API for common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, and more.

## Installation

```bash
pip install -e .
```

After installation, download the required NLTK corpora:

```bash
python -m textblob.download_corpora
```

## Quick Start

```python
from textblob import TextBlob

# Create a TextBlob
blob = TextBlob("TextBlob is amazingly simple to use. What great fun!")

# Get words
print(blob.words)
# WordList(['TextBlob', 'is', 'amazingly', 'simple', 'to', 'use', 'What', 'great', 'fun'])

# Get sentences
print(blob.sentences)
# [Sentence("TextBlob is amazingly simple to use."), Sentence("What great fun!")]

# Sentiment analysis
print(blob.sentiment)
# Sentiment(polarity=0.65, subjectivity=0.75)

# Part-of-speech tags
print(blob.tags)
# [('TextBlob', 'NNP'), ('is', 'VBZ'), ('amazingly', 'RB'), ...]

# Noun phrases
print(blob.noun_phrases)
# WordList(['textblob'])
```

## Features

### Tokenization

```python
from textblob import TextBlob

blob = TextBlob("Hello world. How are you?")
print(blob.words)      # Word tokens
print(blob.sentences)  # Sentence objects
```

### Part-of-Speech Tagging

```python
blob = TextBlob("The quick brown fox jumps over the lazy dog.")
print(blob.tags)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ...]
```

### Sentiment Analysis

```python
blob = TextBlob("This movie is amazing!")
print(blob.sentiment.polarity)      # 0.9 (positive)
print(blob.sentiment.subjectivity)  # 0.85 (subjective)

blob = TextBlob("This movie is terrible.")
print(blob.sentiment.polarity)      # -0.9 (negative)
```

### Word Inflection

```python
from textblob import Word

# Pluralization
word = Word("dog")
print(word.pluralize())  # "dogs"

# Singularization
word = Word("children")
print(word.singularize())  # "child"
```

### Spelling Correction

```python
blob = TextBlob("I havv goood speling!")
print(blob.correct())  # "I have good spelling!"

word = Word("speling")
print(word.correct())  # "spelling"
print(word.spellcheck())  # [("spelling", 0.95), ("spieling", 0.03), ...]
```

### Lemmatization

```python
from textblob import Word

word = Word("running")
print(word.lemmatize("v"))  # "run"

word = Word("better")
print(word.lemmatize("a"))  # "good"
```

### N-grams

```python
blob = TextBlob("The quick brown fox")
print(blob.ngrams(n=2))
# [WordList(['The', 'quick']), WordList(['quick', 'brown']), WordList(['brown', 'fox'])]
```

### WordNet Integration

```python
from textblob import Word

word = Word("happy")
print(word.synsets)      # List of WordNet synsets
print(word.definitions)  # List of definitions
```

### Text Classification

```python
from textblob.classifiers import NaiveBayesClassifier

train = [
    ("I love this product", "pos"),
    ("This is great", "pos"),
    ("Terrible experience", "neg"),
    ("I hate it", "neg"),
]

classifier = NaiveBayesClassifier(train)
print(classifier.classify("This is wonderful!"))  # "pos"
print(classifier.classify("This is awful"))       # "neg"
```

### Blobber Factory

For creating many TextBlobs with the same configuration:

```python
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer

tb = Blobber(analyzer=NaiveBayesAnalyzer())
blob1 = tb("This is great!")
blob2 = tb("This is terrible.")
# Both blobs share the same analyzer instance
```

## API Reference

### TextBlob

The main class for text processing.

**Properties:**
- `words` - WordList of word tokens
- `sentences` - List of Sentence objects
- `tokens` - WordList of all tokens (including punctuation)
- `tags` / `pos_tags` - List of (word, POS tag) tuples
- `noun_phrases` - WordList of noun phrases
- `sentiment` - Sentiment namedtuple (polarity, subjectivity)
- `polarity` - Sentiment polarity (-1.0 to 1.0)
- `subjectivity` - Subjectivity score (0.0 to 1.0)
- `word_counts` - Dictionary of word frequencies
- `np_counts` - Dictionary of noun phrase frequencies

**Methods:**
- `ngrams(n=3)` - Generate n-grams
- `correct()` - Return spell-corrected blob
- `parse()` - Return parsed string
- `classify()` - Classify using configured classifier

### Word

A string subclass with NLP capabilities.

**Methods:**
- `singularize()` - Return singular form
- `pluralize()` - Return plural form
- `lemmatize(pos=None)` - Return lemma
- `stem()` - Return stemmed form
- `spellcheck()` - Return spelling suggestions
- `correct()` - Return corrected word
- `get_synsets(pos=None)` - Get WordNet synsets
- `define(pos=None)` - Get definitions

### WordList

A list subclass for Word collections.

**Methods:**
- `count(word, case_sensitive=False)` - Count occurrences
- `upper()` / `lower()` - Case transformation
- `singularize()` / `pluralize()` - Inflection
- `lemmatize()` / `stem()` - Base forms

## Requirements

- Python >= 3.10
- nltk >= 3.9

## License

MIT License
