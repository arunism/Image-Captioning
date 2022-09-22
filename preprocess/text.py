import re
import string


class CleanImageDescription:
    def __init__(self, datapath: str,):
        self.datapath = datapath

    def read_descriptions(self):
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

    def clean_punctuations(self, sentence):
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)
        sentence = re.sub(r'[?|$|.|!]', r'', sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)
        sentence = re.sub(' +', ' ', sentence)
        return sentence
