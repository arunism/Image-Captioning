import os
from preprocess import CleanImageDescription
from models import ResnetEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    cid = CleanImageDescription(os.path.join(BASE_DIR, 'dataset/captions.txt'))
    desc = cid.clean_descriptions()
    vocab = cid.create_vocab()

    imf = ResnetEncoder('abc')
