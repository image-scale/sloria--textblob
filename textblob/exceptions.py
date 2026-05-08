"""Custom exceptions for TextBlob."""


class TextBlobError(Exception):
    """Base exception for TextBlob errors."""
    pass


class MissingCorpusError(TextBlobError):
    """Exception raised when a required NLTK corpus is not found.

    Attributes:
        corpus_name: Name of the missing corpus
        message: Explanation of the error
    """

    def __init__(self, corpus_name, message=None):
        self.corpus_name = corpus_name
        if message is None:
            message = (
                f"The '{corpus_name}' corpus is required but not found. "
                f"Please download it using: python -m textblob.download_corpora"
            )
        self.message = message
        super().__init__(self.message)


class FormatError(TextBlobError):
    """Exception raised when a file format is invalid or unsupported."""
    pass


class TranslatorError(TextBlobError):
    """Exception raised for translation errors.

    Note: Translation functionality is deprecated but this exception
    is kept for backwards compatibility.
    """
    pass


class NotTranslated(TranslatorError):
    """Exception raised when text could not be translated.

    Note: Translation functionality is deprecated but this exception
    is kept for backwards compatibility.
    """
    pass
