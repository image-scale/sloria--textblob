"""Noun phrase extractors for TextBlob."""

from .base import BaseNPExtractor
from .taggers import NLTKTagger


class FastNPExtractor(BaseNPExtractor):
    """Fast rule-based noun phrase extractor.

    Uses POS tags to identify noun phrases based on patterns like:
    - DT? JJ* NN+  (determiner, adjectives, nouns)
    - NNP+         (proper nouns)
    """

    def __init__(self):
        self._tagger = None

    @property
    def tagger(self):
        if self._tagger is None:
            self._tagger = NLTKTagger()
        return self._tagger

    def extract(self, text):
        """Extract noun phrases from text.

        Args:
            text: The text to extract from

        Returns:
            List of noun phrase strings.
        """
        tagged = self.tagger.tag(text)
        noun_phrases = []
        current_np = []
        in_np = False

        for word, tag in tagged:
            if tag.startswith('NN') or tag == 'JJ' or tag == 'DT':
                if tag.startswith('NN'):
                    in_np = True
                    current_np.append(word)
                elif in_np or tag in ('JJ', 'DT'):
                    current_np.append(word)
            else:
                if in_np and current_np:
                    np = self._clean_np(current_np)
                    if np:
                        noun_phrases.append(np)
                current_np = []
                in_np = False

        if in_np and current_np:
            np = self._clean_np(current_np)
            if np:
                noun_phrases.append(np)

        return noun_phrases

    def _clean_np(self, np_words):
        """Clean up a noun phrase by removing leading determiners."""
        while np_words and np_words[0].lower() in ('the', 'a', 'an', 'this', 'that', 'these', 'those'):
            np_words = np_words[1:]

        if not np_words:
            return None

        has_noun = any(True for w in np_words
                       if not w.lower() in ('the', 'a', 'an', 'very', 'really'))

        if not has_noun:
            return None

        return ' '.join(np_words)


class ConllExtractor(BaseNPExtractor):
    """Chunk-parser based noun phrase extractor.

    Uses NLTK's chunk parser trained on CoNLL-2000 corpus.
    More accurate but slower than FastNPExtractor.
    """

    def __init__(self):
        self._chunker = None

    @property
    def chunker(self):
        if self._chunker is None:
            self._chunker = self._train_chunker()
        return self._chunker

    def _train_chunker(self):
        """Train a chunk parser on CoNLL-2000 data."""
        import nltk
        from nltk.corpus import conll2000

        train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

        class UnigramChunker(nltk.ChunkParserI):
            def __init__(self, train_sents):
                train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                              for sent in train_sents]
                self.tagger = nltk.UnigramTagger(train_data)

            def parse(self, sentence):
                pos_tags = [pos for (word, pos) in sentence]
                tagged_pos_tags = self.tagger.tag(pos_tags)
                chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
                conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                             in zip(sentence, chunktags)]
                return nltk.chunk.conlltags2tree(conlltags)

        return UnigramChunker(train_sents)

    def extract(self, text):
        """Extract noun phrases from text."""
        from .taggers import NLTKTagger
        tagger = NLTKTagger()
        tagged = tagger.tag(text)

        tree = self.chunker.parse(tagged)
        noun_phrases = []

        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                np = ' '.join(word for word, tag in subtree.leaves())
                noun_phrases.append(np)

        return noun_phrases
