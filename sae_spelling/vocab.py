import nltk


def nltk_words(download: bool = True):
    """Get the list of words from the NLTK corpus, downloading if necessary."""
    try:
        return nltk.corpus.words.words()
    except LookupError:
        if not download:
            raise
        nltk.download("words")
    return nltk_words(download=False)
