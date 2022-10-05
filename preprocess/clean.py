import re
import string


class CleanImageDescription:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.descriptions = self.read_descriptions()
        self.vocab = set()

    def read_descriptions(self) -> dict:
        file = open(self.data_path, 'r')
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
