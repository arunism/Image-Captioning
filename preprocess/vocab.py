import os
from collections import Counter
from itertools import islice, chain, repeat
import config
from preprocess import CleanImageDescription

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Vocab:
    def __init__(self):
        self.vocab = set()  # Vocabulary of words
        self.w2i = dict()  # Word to index
        self.i2w = dict()  # Index to word
        self.freq = dict()  # Word frequency
        self.descriptions = None   # Dictionary of descriptions for each image
        self.create_vocab(clean=config.CLEAN_TEXT)
        self.vectorize_descriptions()

    def create_vocab(self, clean=True):
        cid = CleanImageDescription(os.path.join(BASE_DIR, config.TEXT_DATA_PATH))
        self.descriptions = cid.clean_descriptions() if clean else cid.descriptions
        self.freq = dict(Counter(
            word for desc in self.descriptions.values() for sentence in desc for word in sentence.split()
        ))
        self.vocab = {k for k, v in self.freq.items() if v >= config.VOCAB_THRESHOLD}
        self.vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(self.vocab)
        self.w2i = {k: v for v, k in enumerate(self.vocab)}
        self.i2w = {k: v for k, v in enumerate(self.vocab)}

    # def padding(self, sequence):
    #     return list(zip(*zip_longest(*sequence, fillvalue=self.w2i['<PAD>'])))

    def padding(self, sequences):
        sos, eos, pad = [self.w2i['<SOS>']], [self.w2i['<EOS>']], [self.w2i['<PAD>']]
        return [
            sos + sequence[:(config.SEQUENCE_LENGTH - 2)] + eos if len(sequence) > (config.SEQUENCE_LENGTH - 2)
            else sos + sequence + eos + pad*(config.SEQUENCE_LENGTH - len(sequence) - 2)
            for sequence in sequences
        ]

    def vectorize_descriptions(self):
        for filename, descriptions in self.descriptions.items():
            self.descriptions[filename] = [
                [self.w2i.get(word, self.w2i['<UNK>']) for word in description.split()]
                for description in descriptions
            ]
            self.descriptions[filename] = self.padding(self.descriptions[filename])

    def __len__(self):
        return len(self.vocab)
