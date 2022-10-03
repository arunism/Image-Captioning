import torch
import torch.nn as nn
import config
from models import ResnetModel


class Train:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = self.get_optimizer()
        self.model = ResnetModel(1234).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def get_optimizer(self):
        if config.OPTIMIZER.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=config.LR)
        elif config.OPTIMIZER.lower() == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=config.LR)
        elif config.OPTIMIZER.lower() == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=config.LR)
        elif config.OPTIMIZER.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=config.LR)
        elif config.OPTIMIZER.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=config.LR)
        return None

    def train_epoch(self):
        pass

    def train(self):
        pass
