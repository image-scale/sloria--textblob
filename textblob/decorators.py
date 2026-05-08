"""Decorators for TextBlob."""

import functools
from .exceptions import MissingCorpusError


class cached_property:
    """A property that is only computed once and then cached.

    Similar to functools.cached_property but compatible with older Python.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__
        self.attr_name = None

    def __set_name__(self, owner, name):
        self.attr_name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attr_name is None:
            self.attr_name = self.func.__name__
        cache_attr = f"_cached_{self.attr_name}"
        if not hasattr(instance, cache_attr):
            setattr(instance, cache_attr, self.func(instance))
        return getattr(instance, cache_attr)


def requires_nltk_corpus(corpus_name):
    """Decorator that ensures an NLTK corpus is available before running.

    Args:
        corpus_name: Name of the required NLTK corpus

    Raises:
        MissingCorpusError: If the corpus is not found
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import nltk
                nltk.data.find(f"corpora/{corpus_name}")
            except LookupError:
                try:
                    nltk.data.find(f"tokenizers/{corpus_name}")
                except LookupError:
                    try:
                        nltk.data.find(f"taggers/{corpus_name}")
                    except LookupError:
                        raise MissingCorpusError(corpus_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator
