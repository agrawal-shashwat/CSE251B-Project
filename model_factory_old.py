################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
import torch.nn as nn
import torchvision

class CaptionModel(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size):
        super(CaptionModel, self).__init__()
        
        # Load pretrained ResNet50 and remove last FC layer
        self.resnet = nn.Sequential(*list(torchvision.models.resnet101(pretrained = True).children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.h_init = nn.Linear(2048, hidden_size)
        self.c_init = nn.Linear(2048, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = 2, batch_first = True, dropout = 0.5)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, y, state = None):        
        # Compute Embedding
        if state is None:
            emb = self.embedding(y[:, :-1])
        elif state == 'init':
            emb = self.embedding(y[:, 0][:, None])
        else:
            emb = self.embedding(x)
        
        # Run encoder
        if state is None or state == 'init':
            x = self.resnet(x)
            x = x[..., 0, 0] 
            h = self.h_init(x).repeat(2, 1, 1)
            c = self.c_init(x).repeat(2, 1, 1)
            state = (h, c)
        
        # LSTM
        out, state = self.lstm(emb, state)
        out = self.fc_out(out)
        out = out.permute(0, 2, 1)
        return out, state
        
class vanilla_RNN_Model(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size):
        super(vanilla_RNN_Model, self).__init__()
        
        # Load pretrained ResNet50 and remove last FC layer
        self.resnet = nn.Sequential(*list(torchvision.models.resnet50(pretrained = True).children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.h_init = nn.Linear(2048, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx = 0)
        self.rnn = nn.RNN(embedding_size, hidden_size, num_layers = 2, nonlinearity = 'relu', batch_first = True, dropout = 0.5)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, y, state = None):
        # Compute Encoding
        if state is None:
            emb = self.embedding(y[:, :-1])
        elif state == 'init':
            emb = self.embedding(y[:, 0][:, None])
        else:
            emb = self.embedding(x)
            
        # Run encoder
        if state is None or state == 'init':
            x = self.resnet(x)
            x = x[..., 0, 0] 
            state = self.h_init(x).repeat(2, 1, 1)
        
        # RNN
        out, state = self.rnn(emb, state)
        out = self.fc_out(out)
        out = out.permute(0, 2, 1)
        return out, state
    
# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    if model_type == 'LSTM':
        return CaptionModel(hidden_size, embedding_size, vocab.idx)
    elif model_type == 'vanilla_RNN':
        return vanilla_RNN_Model(hidden_size, embedding_size, vocab.idx)
