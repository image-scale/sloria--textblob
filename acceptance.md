# Acceptance Criteria

## Task 1: Core TextBlob class with tokenization and string-like behavior

### Acceptance Criteria
- [ ] TextBlob("Hello world") creates a text blob object
- [ ] TextBlob must be initialized with a string, raises TypeError for non-strings
- [ ] TextBlob.raw returns the original text string
- [ ] TextBlob.words returns a list of word tokens (excluding punctuation)
- [ ] TextBlob.tokens returns all tokens including punctuation
- [ ] TextBlob.sentences returns a list of Sentence objects
- [ ] TextBlob.raw_sentences returns a list of raw sentence strings
- [ ] len(TextBlob("hello")) returns 5 (character count)
- [ ] TextBlob supports iteration over characters
- [ ] TextBlob supports "in" operator for substring checking
- [ ] TextBlob supports indexing and slicing (blob[0], blob[0:5])
- [ ] TextBlob supports string methods: upper(), lower(), strip(), title(), find(), rfind(), startswith(), endswith(), replace(), join(), format(), split(), index()
- [ ] TextBlob supports comparison operators (==, <, >, <=, >=, !=) with strings and other blobs
- [ ] TextBlob supports concatenation with + operator (blob + blob, blob + string)
- [ ] TextBlob supports hashing (hash(blob))
- [ ] repr(TextBlob("text")) returns 'TextBlob("text")'
- [ ] str(TextBlob("text")) returns "text"
- [ ] TextBlob.ngrams(n=3) returns list of n-word tuples from the text
- [ ] ngrams(n=0) or ngrams(n<0) returns empty list
- [ ] WordTokenizer.tokenize(text) returns list of word tokens
- [ ] WordTokenizer.tokenize(text, include_punc=False) excludes punctuation
- [ ] SentenceTokenizer.tokenize(text) returns list of sentences
- [ ] sent_tokenize(text) is a generator that yields sentences
- [ ] word_tokenize(text) is a generator that yields words
- [ ] Tokenizers have itokenize() method that returns a generator
