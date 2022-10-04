import os
from torch.utils.data import DataLoader

import config
from preprocess import ImageCaptionDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    icd = ImageCaptionDataset()
    data_loader = DataLoader(dataset=icd, batch_size=config.BATCH_SIZE, shuffle=True)
    print(len(data_loader[0]))
