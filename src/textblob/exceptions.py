"""Custom exceptions for the textblob library."""

MISSING_CORPUS_MESSAGE = """
Looks like you are missing some required data for this feature.

To download the necessary data, simply run

    python -m textblob.download_corpora

or use the NLTK downloader to download the missing data: http://nltk.org/data.html
"""


class TextBlobError(Exception):
    """A TextBlob-related error."""
    pass


class MissingCorpusError(TextBlobError):
    """Exception thrown when a user tries to use a feature that requires a
    dataset or model that the user does not have on their system.
    """

    def __init__(self, message=MISSING_CORPUS_MESSAGE, *args, **kwargs):
        super().__init__(message, *args, **kwargs)


class FormatError(TextBlobError):
    """Raised if a data file with an unsupported format is passed to a classifier."""
    pass
