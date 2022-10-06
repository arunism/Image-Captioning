import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from preprocess import Vocab
import config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImageCaptionDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.images_dir = os.path.join(BASE_DIR, config.IMAGE_DATA_PATH)
        self.captions = list(Vocab().descriptions.items())

    def __getitem__(self, idx):
        image = read_image(os.path.join(self.images_dir, self.captions[idx][0])).float()
        image = self.transform(image) if self.transform else image
        captions = torch.LongTensor(self.captions[idx][1])
        return image, captions

    def __len__(self):
        return len(self.captions)
