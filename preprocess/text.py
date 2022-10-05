import re
import string

import config


class CleanImageDescription:
    def __init__(self, datapath: str):
        self.datapath = datapath
        self.descriptions = self.read_descriptions()
        self.vocab = set()
        self.descriptions = self.clean_descriptions()
        self.vocab = self.create_vocab()
        print(self.vocab)

    def read_descriptions(self) -> dict:
        file = open(self.datapath, 'r')
        text = file.read()
        file.close()

        descriptions = dict()
        for line in text.split('\n'):
            if len(line) < 20:
                continue
            tokens = line.split(',', 1)
            filename, desc = tokens[0], tokens[1:]
            if filename not in descriptions:
                descriptions[filename] = list()
            descriptions[filename].append(desc[0])
        return descriptions

    def clean_descriptions(self) -> dict:
        for filename, descriptions in self.descriptions.items():
            for i in range(len(descriptions)):
                description = descriptions[i]
                table = str.maketrans('', '', string.punctuation)
                description = description.translate(table).lower()
                description = re.sub(' +', ' ', description).lstrip().rstrip()
                descriptions[i] = description
        return self.descriptions

    def create_vocab(self) -> dict:
        for descriptions in self.descriptions.values():
            [self.vocab.update(description.split()) for description in descriptions]
        self.vocab = ['<PAD>', '<UNK>'] + sorted(self.vocab)
        return {k: v for v, k in enumerate(self.vocab)}

    def padding(self, text):
        text = [
            sentence[:config.SEQUENCE_LENGTH] if len(sentence) > config.SEQUENCE_LENGTH
            else sentence + [0]*(config.SEQUENCE_LENGTH - len(sentence))
            for sentence in text
        ]
        return text

    def vector_descriptions(self, clean=True) -> dict:
        self.vocab = self.create_vocab()
        self.descriptions = self.clean_descriptions(self.descriptions) if clean else self.descriptions
        vector = dict()
        for filename, descriptions in self.descriptions.items():
            vector[filename] = [
                [self.vocab.get(word, self.vocab['<UNK>']) for word in description]
                for description in descriptions
            ]

