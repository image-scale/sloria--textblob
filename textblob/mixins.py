"""Mixin classes for TextBlob."""

import functools


@functools.total_ordering
class ComparableMixin:
    """Mixin class that defines comparison operations.

    Subclasses must define _cmpkey() which returns a comparable key.
    """

    def _cmpkey(self):
        """Return a key for comparisons. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _cmpkey()")

    def __eq__(self, other):
        if isinstance(other, ComparableMixin):
            return self._cmpkey() == other._cmpkey()
        return self._cmpkey() == other

    def __lt__(self, other):
        if isinstance(other, ComparableMixin):
            return self._cmpkey() < other._cmpkey()
        return self._cmpkey() < other

    def __hash__(self):
        return hash(self._cmpkey())


class BlobComparableMixin(ComparableMixin):
    """Mixin for comparing blob objects by their raw text."""

    def _cmpkey(self):
        return self.raw

    def __eq__(self, other):
        if isinstance(other, str):
            return self.raw == other
        if isinstance(other, BlobComparableMixin):
            return self.raw == other.raw
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __lt__(self, other):
        if isinstance(other, str):
            return self.raw < other
        if isinstance(other, BlobComparableMixin):
            return self.raw < other.raw
        return NotImplemented

    def __le__(self, other):
        return self == other or self < other

    def __gt__(self, other):
        if isinstance(other, str):
            return self.raw > other
        if isinstance(other, BlobComparableMixin):
            return self.raw > other.raw
        return NotImplemented

    def __ge__(self, other):
        return self == other or self > other

    def __hash__(self):
        return hash(self.raw)


class StringlikeMixin:
    """Mixin that makes an object behave like a string.

    The object must have a `raw` property that returns the underlying string.
    """

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(\"{self.raw}\")"

    def __str__(self):
        return self.raw

    def __len__(self):
        return len(self.raw)

    def __iter__(self):
        return iter(self.raw)

    def __contains__(self, item):
        return item in self.raw

    def __getitem__(self, key):
        return self.raw[key]

    def __add__(self, other):
        if isinstance(other, str):
            return self.__class__(self.raw + other)
        if hasattr(other, 'raw'):
            return self.__class__(self.raw + other.raw)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, str):
            return self.__class__(other + self.raw)
        return NotImplemented

    def __mul__(self, n):
        return self.__class__(self.raw * n)

    def __rmul__(self, n):
        return self.__class__(n * self.raw)

    def find(self, sub, start=0, end=None):
        if end is None:
            return self.raw.find(sub, start)
        return self.raw.find(sub, start, end)

    def rfind(self, sub, start=0, end=None):
        if end is None:
            return self.raw.rfind(sub, start)
        return self.raw.rfind(sub, start, end)

    def index(self, sub, start=0, end=None):
        if end is None:
            return self.raw.index(sub, start)
        return self.raw.index(sub, start, end)

    def rindex(self, sub, start=0, end=None):
        if end is None:
            return self.raw.rindex(sub, start)
        return self.raw.rindex(sub, start, end)

    def startswith(self, prefix, start=0, end=None):
        if end is None:
            return self.raw.startswith(prefix, start)
        return self.raw.startswith(prefix, start, end)

    def endswith(self, suffix, start=0, end=None):
        if end is None:
            return self.raw.endswith(suffix, start)
        return self.raw.endswith(suffix, start, end)

    def title(self):
        return self.__class__(self.raw.title())

    def format(self, *args, **kwargs):
        return self.raw.format(*args, **kwargs)

    def split(self, sep=None, maxsplit=-1):
        return self.raw.split(sep, maxsplit)

    def strip(self, chars=None):
        return self.__class__(self.raw.strip(chars))

    def lstrip(self, chars=None):
        return self.__class__(self.raw.lstrip(chars))

    def rstrip(self, chars=None):
        return self.__class__(self.raw.rstrip(chars))

    def upper(self):
        return self.__class__(self.raw.upper())

    def lower(self):
        return self.__class__(self.raw.lower())

    def join(self, iterable):
        return self.raw.join(iterable)

    def replace(self, old, new, count=-1):
        return self.__class__(self.raw.replace(old, new, count))

    def capitalize(self):
        return self.__class__(self.raw.capitalize())

    def swapcase(self):
        return self.__class__(self.raw.swapcase())

    def center(self, width, fillchar=' '):
        return self.__class__(self.raw.center(width, fillchar))

    def count(self, sub, start=0, end=None):
        if end is None:
            return self.raw.count(sub, start)
        return self.raw.count(sub, start, end)

    def encode(self, encoding='utf-8', errors='strict'):
        return self.raw.encode(encoding, errors)

    def expandtabs(self, tabsize=8):
        return self.__class__(self.raw.expandtabs(tabsize))

    def isalnum(self):
        return self.raw.isalnum()

    def isalpha(self):
        return self.raw.isalpha()

    def isdigit(self):
        return self.raw.isdigit()

    def islower(self):
        return self.raw.islower()

    def isspace(self):
        return self.raw.isspace()

    def istitle(self):
        return self.raw.istitle()

    def isupper(self):
        return self.raw.isupper()

    def ljust(self, width, fillchar=' '):
        return self.__class__(self.raw.ljust(width, fillchar))

    def rjust(self, width, fillchar=' '):
        return self.__class__(self.raw.rjust(width, fillchar))

    def partition(self, sep):
        return self.raw.partition(sep)

    def rpartition(self, sep):
        return self.raw.rpartition(sep)

    def splitlines(self, keepends=False):
        return self.raw.splitlines(keepends)

    def zfill(self, width):
        return self.__class__(self.raw.zfill(width))
