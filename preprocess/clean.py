import re
import string


class CleanImageDescription:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.descriptions = self.read_descriptions()
        self.vocab = set()

    def read_descriptions(self) -> list:
        file = open(self.data_path, 'r')
        text = file.read()
        file.close()

        descriptions = list()
        for line in text.split('\n'):
            tokens = line.split(',', 1)
            filename, desc = tokens[0], ', '.join(tokens[1:])
            if not filename.endswith(('.jpg', '.jpeg', '.png')):
                continue
            descriptions.append((filename, desc))
        return descriptions

    def clean_descriptions(self) -> list:
        descriptions = list()
        for filename, description in self.descriptions:
            table = str.maketrans('', '', string.punctuation)
            description = description.translate(table).lower()
            description = re.sub(' +', ' ', description).lstrip().rstrip()
            descriptions.append((filename, description))
        self.descriptions = descriptions
        return self.descriptions
