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
        self.descriptions = None   # Dictionary of descriptions for each image
        self.create_vocab(clean=config.CLEAN_TEXT)
        self.vectorize_descriptions()

    def create_vocab(self, clean=True):
        cid = CleanImageDescription(os.path.join(BASE_DIR, config.TEXT_DATA_PATH))
        self.descriptions = cid.clean_descriptions() if clean else cid.descriptions
        self.freq = dict(Counter(
            word for description in self.descriptions for word in description[1].split()
        ))
        self.vocab = {k for k, v in self.freq.items() if v >= config.VOCAB_THRESHOLD}
        self.vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + sorted(self.vocab)
        self.w2i = {k: v for v, k in enumerate(self.vocab)}
        self.i2w = {k: v for k, v in enumerate(self.vocab)}

    # def padding(self, sequence):
    #     return list(zip(*zip_longest(*sequence, fillvalue=self.w2i['<PAD>'])))

    def padding(self, sequence):
        sos, eos, pad = [self.w2i['<SOS>']], [self.w2i['<EOS>']], [self.w2i['<PAD>']]
        return (
            sos + sequence[:(config.SEQUENCE_LENGTH - 2)] + eos if len(sequence) > (config.SEQUENCE_LENGTH - 2)
            else sos + sequence + eos + pad * (config.SEQUENCE_LENGTH - len(sequence) - 2)
        )

    def vectorize_descriptions(self):
        descriptions = list()
        for filename, description in self.descriptions:
            vector = [self.w2i.get(word, self.w2i['<UNK>']) for word in description.split()]
            padded_vector = self.padding(vector)
            descriptions.append((filename, padded_vector))
        self.descriptions = descriptions

    def __len__(self):
        return len(self.vocab)
