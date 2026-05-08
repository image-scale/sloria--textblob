"""Text parsers for TextBlob."""

from .base import BaseParser
from .taggers import NLTKTagger


class PatternParser(BaseParser):
    """Parser that returns tagged and chunked text.

    Returns text in the format: word/POS/CHUNK
    """

    def __init__(self):
        self._tagger = None

    @property
    def tagger(self):
        if self._tagger is None:
            self._tagger = NLTKTagger()
        return self._tagger

    def parse(self, text):
        """Parse text and return tagged string.

        Args:
            text: The text to parse

        Returns:
            String with words tagged: "word/TAG/CHUNK ..."
        """
        tagged = self.tagger.tag(text)
        chunks = self._chunk(tagged)

        result = []
        for word, pos, chunk in chunks:
            result.append(f"{word}/{pos}/{chunk}")

        return ' '.join(result)

    def _chunk(self, tagged):
        """Add chunk tags to tagged words.

        Args:
            tagged: List of (word, pos) tuples

        Returns:
            List of (word, pos, chunk) tuples
        """
        chunked = []
        in_np = False

        for i, (word, pos) in enumerate(tagged):
            if pos.startswith('NN') or pos == 'PRP':
                if not in_np:
                    chunk = 'B-NP'
                    in_np = True
                else:
                    chunk = 'I-NP'
            elif pos.startswith('JJ') or pos == 'DT':
                if i + 1 < len(tagged) and tagged[i + 1][1].startswith('NN'):
                    if not in_np:
                        chunk = 'B-NP'
                        in_np = True
                    else:
                        chunk = 'I-NP'
                else:
                    chunk = 'O'
                    in_np = False
            elif pos.startswith('VB'):
                chunk = 'B-VP'
                in_np = False
            elif pos.startswith('RB'):
                chunk = 'B-ADVP'
                in_np = False
            elif pos in ('IN', 'TO'):
                chunk = 'B-PP'
                in_np = False
            else:
                chunk = 'O'
                in_np = False

            chunked.append((word, pos, chunk))

        return chunked
