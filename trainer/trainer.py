import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import config
from preprocess import Vocab
from models import ResnetModel
from data import ImageCaptionDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = len(Vocab())
        self.model = ResnetModel(self.vocab_size).to(self.device)
        self.optimizer = self.get_optimizer()
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.result_dir = os.path.join(BASE_DIR, config.OUTPUT_PATH)
        if not os.path.exists(self.result_dir): os.mkdir(self.result_dir)

    def get_optimizer(self):
        if config.OPTIMIZER.lower() == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=config.LR)
        elif config.OPTIMIZER.lower() == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=config.LR)
        elif config.OPTIMIZER.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=config.LR)
        elif config.OPTIMIZER.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=config.LR)
        return torch.optim.Adam(self.model.parameters(), lr=config.LR)

    def train_epoch(self, data_loader):
        for image, captions in tqdm(data_loader):
            image, captions = image.to(self.device), captions.to(self.device)
            output = self.model(image, captions)
            loss = self.criterion(output.view(-1, output.shape[2]), captions.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print('LOSS: ', loss, end=' ')
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(checkpoint, os.path.join(self.result_dir, 'model.pth.tar'))

    def train(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        data = ImageCaptionDataset(transform=transform)
        data_loader = DataLoader(dataset=data, batch_size=config.BATCH_SIZE, shuffle=True)
        self.model.train()
        for epoch in range(config.EPOCHS):
            self.train_epoch(data_loader)
