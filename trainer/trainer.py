import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import config
from preprocess import Vocab
from models import ResnetModel
from data import ImageCaptionDataset


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = len(Vocab())
        self.model = ResnetModel(self.vocab_size).to(self.device)
        self.optimizer = self.get_optimizer()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

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
        pass

    def train(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        data = ImageCaptionDataset(transform=transform)
        data_loader = DataLoader(dataset=data, batch_size=config.BATCH_SIZE, shuffle=True)
        self.model.train()
        self.train_epoch(data_loader)
