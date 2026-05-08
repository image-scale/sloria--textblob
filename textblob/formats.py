"""File format handlers for classifier training data."""

import json
import csv
from io import StringIO


class BaseFormat:
    """Base class for file format handlers."""

    @classmethod
    def detect(cls, stream):
        """Detect if stream is in this format."""
        return False

    @classmethod
    def read(cls, stream):
        """Read training data from stream.

        Returns:
            List of (text, label) tuples
        """
        raise NotImplementedError


class CSV(BaseFormat):
    """CSV format handler.

    Expected format: text,label (one per line)
    """

    @classmethod
    def detect(cls, stream):
        try:
            content = stream.read(1024)
            stream.seek(0)
            return ',' in content and not content.strip().startswith('{')
        except:
            return False

    @classmethod
    def read(cls, stream):
        content = stream.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        result = []
        reader = csv.reader(StringIO(content))
        for row in reader:
            if len(row) >= 2:
                text = row[0]
                label = row[-1]
                result.append((text, label))
        return result


class TSV(BaseFormat):
    """TSV (tab-separated) format handler.

    Expected format: label\ttext (one per line)
    """

    @classmethod
    def detect(cls, stream):
        try:
            content = stream.read(1024)
            stream.seek(0)
            return '\t' in content and ',' not in content
        except:
            return False

    @classmethod
    def read(cls, stream):
        content = stream.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        result = []
        for line in content.strip().split('\n'):
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    label, text = parts
                    result.append((text.strip(), label.strip()))
        return result


class JSON(BaseFormat):
    """JSON format handler.

    Expected format: [{"text": "...", "label": "..."}, ...]
    """

    @classmethod
    def detect(cls, stream):
        try:
            content = stream.read(1024)
            stream.seek(0)
            return content.strip().startswith('[') or content.strip().startswith('{')
        except:
            return False

    @classmethod
    def read(cls, stream):
        content = stream.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        data = json.loads(content)
        if isinstance(data, list):
            return [(item['text'], item['label']) for item in data]
        else:
            return [(data['text'], data['label'])]


_registry = {
    'csv': CSV,
    'tsv': TSV,
    'json': JSON,
}


def register(name, format_class):
    """Register a custom format handler.

    Args:
        name: Format name (e.g., 'psv' for pipe-separated)
        format_class: Format handler class
    """
    _registry[name.lower()] = format_class


def get_format(name):
    """Get a format handler by name.

    Args:
        name: Format name

    Returns:
        Format handler class

    Raises:
        KeyError: If format not found
    """
    return _registry[name.lower()]


def detect_format(stream):
    """Detect the format of a stream.

    Args:
        stream: File-like object

    Returns:
        Format handler class or None
    """
    for format_class in _registry.values():
        if format_class.detect(stream):
            return format_class
    return None
