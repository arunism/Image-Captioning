import torch.nn as nn
from torchvision import models
import config


class ResnetEncoder:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.input_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(self.input_feats, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, directory):
        features = self.model(directory)
        return self.dropout(features)
