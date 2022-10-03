import os
from zipfile import ZipFile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    file = ZipFile(os.path.join(BASE_DIR, 'archive.zip'))
    file.extractall(BASE_DIR)
    file.close()
