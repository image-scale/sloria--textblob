"""Download required NLTK corpora for TextBlob."""

import sys


REQUIRED_CORPORA = [
    'punkt_tab',
    'averaged_perceptron_tagger_eng',
    'wordnet',
    'brown',
]

OPTIONAL_CORPORA = [
    'movie_reviews',
    'conll2000',
]


def download_corpora(all_corpora=False):
    """Download required NLTK corpora.

    Args:
        all_corpora: If True, also download optional corpora
    """
    import nltk

    corpora = REQUIRED_CORPORA.copy()
    if all_corpora:
        corpora.extend(OPTIONAL_CORPORA)

    for corpus in corpora:
        print(f"Downloading {corpus}...")
        try:
            nltk.download(corpus, quiet=True)
            print(f"  [OK] {corpus}")
        except Exception as e:
            print(f"  [FAILED] {corpus}: {e}")

    print("\nDownload complete!")


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download required NLTK corpora for TextBlob'
    )
    parser.add_argument(
        '-a', '--all',
        action='store_true',
        help='Download all corpora including optional ones'
    )
    args = parser.parse_args()

    download_corpora(all_corpora=args.all)


if __name__ == '__main__':
    main()
