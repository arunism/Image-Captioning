import torch
import torch.nn as nn
from torchvision import models
import config


class ResnetEncoder(nn.Module):
    def __init__(self):
        super(ResnetEncoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.input_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(self.input_feats, config.EMBED_SIZE)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, images):
        features = self.model(images)
        return self.dropout(features)


class ResnetDecoder(nn.Module):
    def __init__(self, vocab_size):
        super(ResnetDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, config.EMBED_SIZE)
        self.lstm = nn.LSTM(config.EMBED_SIZE, config.HIDDEN_SIZE)
        self.fc = nn.Linear(config.HIDDEN_SIZE, vocab_size)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, captions, features):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hidden, _ = self.lstm(embeddings)
        output = self.fc(hidden)
        return output


class ResnetModel(nn.Module):
    def __init__(self, vocab_size):
        super(ResnetModel, self).__init__()
        self.encoder = ResnetEncoder()
        self.decoder = ResnetDecoder(vocab_size)

    def forward(self, captions, images):
        features = self.encoder(images)
        outputs = self.decoder(captions, features)
        return outputs
