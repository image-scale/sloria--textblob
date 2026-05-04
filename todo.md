# Todo

## Plan
Implement the textblob library starting with the core TextBlob class that users interact with directly. This class needs tokenizers, taggers, and sentiment analyzers, so we'll build those components alongside the main feature that uses them. Then add Word class with inflection/WordNet features, followed by classifiers and noun phrase extractors.

## Tasks
- [>] Task 1: Implement the core TextBlob class with sentence/word tokenization, basic string-like behavior, and n-grams (src/textblob/blob.py, tokenizers.py, base classes, mixins, utils + tests)
- [ ] Task 2: Implement POS tagging for TextBlob using NLTK tagger (taggers.py, update blob.py + tests)
- [ ] Task 3: Implement sentiment analysis with Pattern-style analyzer that returns polarity and subjectivity scores (sentiments.py, update blob.py + tests)
- [ ] Task 4: Implement Word class with singularize/pluralize, spelling correction, lemmatization, stemming, and WordNet integration (inflect.py, update blob.py + tests)
- [ ] Task 5: Implement WordList class with bulk word operations and case-insensitive counting (update blob.py + tests)
- [ ] Task 6: Implement Sentence class with start/end indices and JSON serialization (update blob.py + tests)
- [ ] Task 7: Implement Blobber factory class for creating TextBlobs with shared settings (update blob.py + tests)
- [ ] Task 8: Implement noun phrase extraction using NLTK chunking (np_extractors.py, update blob.py + tests)
- [ ] Task 9: Implement text classifiers (NaiveBayes, DecisionTree) with file format support for training data (classifiers.py, formats.py + tests)
- [ ] Task 10: Implement NaiveBayes sentiment analyzer that classifies text as positive/negative (update sentiments.py + tests)
- [ ] Task 11: Implement parsing functionality with Pattern-style parser (parsers.py, update blob.py + tests)
