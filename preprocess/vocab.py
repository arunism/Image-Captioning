import os
from collections import Counter
import config
from preprocess import CleanImageDescription

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Vocab:
    def __init__(self):
        self.vocab = set()  # Vocabulary of words
        self.w2i = dict()  # Word to index
        self.i2w = dict()  # Index to word
        self.freq = dict()  # Word frequency
        self.create_vocab(clean=config.CLEAN_TEXT)

    def create_vocab(self, clean=True):
        cid = CleanImageDescription(os.path.join(BASE_DIR, config.TEXT_DATA_PATH))
        descriptions = cid.clean_descriptions() if clean else cid.descriptions
        self.freq = dict(Counter(
            word for desc in descriptions.values() for sentence in desc for word in sentence.split()
        ))
        self.vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(self.freq.keys())
        self.w2i = {k: v for v, k in enumerate(self.vocab)}
        self.i2w = {k: v for k, v in enumerate(self.vocab)}
