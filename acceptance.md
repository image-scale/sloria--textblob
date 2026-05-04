# Acceptance Criteria

## Task 1: Core TextBlob class with tokenization and string-like behavior

### Acceptance Criteria
- [x] TextBlob("Hello world") creates a text blob object
- [x] TextBlob must be initialized with a string, raises TypeError for non-strings
- [x] TextBlob.raw returns the original text string
- [x] TextBlob.words returns a list of word tokens (excluding punctuation)
- [x] TextBlob.tokens returns all tokens including punctuation
- [x] TextBlob.sentences returns a list of Sentence objects
- [x] TextBlob.raw_sentences returns a list of raw sentence strings
- [x] len(TextBlob("hello")) returns 5 (character count)
- [x] TextBlob supports iteration over characters
- [x] TextBlob supports "in" operator for substring checking
- [x] TextBlob supports indexing and slicing (blob[0], blob[0:5])
- [x] TextBlob supports string methods: upper(), lower(), strip(), title(), find(), rfind(), startswith(), endswith(), replace(), join(), format(), split(), index()
- [x] TextBlob supports comparison operators (==, <, >, <=, >=, !=) with strings and other blobs
- [x] TextBlob supports concatenation with + operator (blob + blob, blob + string)
- [x] TextBlob supports hashing (hash(blob))
- [x] repr(TextBlob("text")) returns 'TextBlob("text")'
- [x] str(TextBlob("text")) returns "text"
- [x] TextBlob.ngrams(n=3) returns list of n-word tuples from the text
- [x] ngrams(n=0) or ngrams(n<0) returns empty list
- [x] WordTokenizer.tokenize(text) returns list of word tokens
- [x] WordTokenizer.tokenize(text, include_punc=False) excludes punctuation
- [x] SentenceTokenizer.tokenize(text) returns list of sentences
- [x] sent_tokenize(text) is a generator that yields sentences
- [x] word_tokenize(text) is a generator that yields words
- [x] Tokenizers have itokenize() method that returns a generator

## Task 2: POS tagging for TextBlob

### Acceptance Criteria
- [x] TextBlob.pos_tags returns list of (word, tag) tuples
- [x] TextBlob.tags is an alias for pos_tags
- [x] POS tags use Penn Treebank tagset (e.g., 'NN' for noun, 'VB' for verb)
- [x] NLTKTagger class implements BaseTagger interface
- [x] NLTKTagger.tag(text) returns list of (word, tag) tuples
- [x] POS tagging excludes punctuation from results
- [x] Sentence objects also have pos_tags property
- [x] Can pass custom pos_tagger to TextBlob constructor
- [x] pos_tagger is validated - raises ValueError for invalid tagger

## Task 3: Sentiment analysis with Pattern-style analyzer

### Acceptance Criteria
- [ ] PatternAnalyzer class implements BaseSentimentAnalyzer interface
- [ ] PatternAnalyzer.analyze(text) returns named tuple (polarity, subjectivity)
- [ ] polarity is a float between -1.0 (negative) and 1.0 (positive)
- [ ] subjectivity is a float between 0.0 (objective) and 1.0 (subjective)
- [ ] PatternAnalyzer has kind=CONTINUOUS
- [ ] TextBlob.sentiment returns (polarity, subjectivity) tuple
- [ ] TextBlob.polarity returns polarity score
- [ ] TextBlob.subjectivity returns subjectivity score
- [ ] Positive text has polarity > 0
- [ ] Negative text has polarity < 0
- [ ] Sentence objects also have sentiment/polarity/subjectivity properties
- [ ] Can pass custom analyzer to TextBlob constructor
- [ ] analyzer is validated - raises ValueError for invalid analyzer
