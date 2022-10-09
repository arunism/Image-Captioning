import torch
import torch.nn as nn
import config
from preprocess import Vocab
from models import ResnetModel


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
