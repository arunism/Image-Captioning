import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from preprocess import CleanImageDescription
import config

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ImageCaptionDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.images_dir = os.path.join(BASE_DIR, config.IMAGE_DATA_PATH)
        self.captions_path = os.path.join(BASE_DIR, config.TEXT_DATA_PATH)
        self.captions = CleanImageDescription(self.captions_path).clean_descriptions().items()

    def __getitem__(self, idx):
        image = Image(os.path.join(self.images_dir, self.captions.iloc[idx, 0]))
        image = self.transform(image) if self.transform else image
        captions = torch.LongTensor(self.captions.iloc[idx, 1])
        print(image)
        # print(captions)

    def __len__(self):
        return len(self.captions)
