################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
import math

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
    
class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        

    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features

#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)
        
        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)
        
        
        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)
        
        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)
        
        return alpha,attention_weights
            
#Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,config, vocabulary, drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        attn = BertSelfAttention(config, vocabulary)
        embed_rand= torch.rand((1,3,4))
        self.attention_bert= attn(embed_rand)
        
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        
        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)
        weight = torch.load('./pretrain_weights.pt')
        weight = weight.cuda()
        self.prembed = nn.Embedding.from_pretrained(weight, freeze=True)
        
        
    
    def forward(self, features, captions, pretrained = True):
        if pretrained:
            embeds = self.prembed(captions).float()
        else:
            embeds = self.embedding(captions)
        
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        
       
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).cuda()
        alphas = torch.zeros(batch_size, seq_length,num_features).cuda()
                
        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
                    
            output = self.fcn(self.drop(h))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas
    
    def generate_caption(self,features,config_data, vocab=None, pretrained=True):
        # Inference part
        # Given the image features generate the captions
        max_len=20
        temperature = 0.1
        
        max_len = config_data['generation']['max_length']
        temperature = config_data['generation']['temperature']
        isDeterministic = config_data['generation']['deterministic']
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        
        alphas = []
        
        #starting input
        word = torch.tensor(vocab.word2idx['<start>']).view(1,-1).cuda()
        if pretrained:
            embeds = self.prembed(word).float()
        else:
            embeds = self.embedding(word)
       

        
        captions = []
        
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            
            
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)
        
            
            #select the word with most val
            #predicted_word_idx = output.argmax(dim=1)
            
            if isDeterministic == True:
                output = output.view(batch_size,-1)
                 #select the word with most val
                predicted_word_idx = output.argmax(dim=1)
                
            else:           
                pred = output #.data.cpu().numpy()[-1, :]
                pred = F.softmax(pred / temperature, dim = -1)              
                # random sample from the predicted distribution
                predicted_word_idx = torch.multinomial(pred, 1) 

            #save the generated word
            captions.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if vocab.idx2word[predicted_word_idx.item()] == "<end>":
                break
            
            
            #send generated word as the next caption
           # embeds = self.embedding(predicted_word_idx.unsqueeze(0))
            if isDeterministic == True:
                predicted_word_idx = predicted_word_idx.unsqueeze(0)
                
            if pretrained:
                embeds = self.prembed(predicted_word_idx).float()
            else:
                embeds = self.embedding(predicted_word_idx)
        
        #covert the vocab idx to words and return sentence
        return [vocab.idx2word[idx] for idx in captions],alphas
    
    
    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c
    
class transformer_Model(nn.Module):
    def __init__(self,config_data, vocab):
        super().__init__()
        embed_size = config_data['model']['embedding_size']
        vocab_size = vocab.idx
        attention_dim = config_data['model']['attention_size']
        encoder_dim = config_data['model']['encoder_size']
        decoder_dim = config_data['model']['decoder_size']
        drop_prob=0.3
        self.encoder = EncoderCNN()#replace with image and question encoder
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim, 
            config= config_data,
            vocabulary= vocab
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)#cat both image and the question
        outputs = self.decoder(features, captions)
        return outputs
    
    

# #Muti Headed Attention
class BertSelfAttention(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()

        self.num_attention_heads = config['model']['num_of_attention_heads']
        self.attention_head_size = int(config['model']['hidden_size'] / config['model']['num_of_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['model']['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['model']['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['model']['hidden_size'], self.all_head_size)

        self.dense = nn.Linear(config['model']['hidden_size'], config['model']['hidden_size'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        #return x

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)                             # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)                                 # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)                             # [Batch_size x Seq_length x Hidden_size]
        
        query_layer = self.transpose_for_scores(mixed_query_layer)                # [Batch_size x Num_of_heads x Seq_length x Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)                    # [Batch_size x Num_of_heads x Seq_length x Head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer)                # [Batch_size x Num_of_heads x Seq_length x Head_size]

        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)                    # [Batch_size x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs, value_layer)                # [Batch_size x Num_of_heads x Seq_length x Head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()            # [Batch_size x Seq_length x Num_of_heads x Head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [Batch_size x Seq_length x Hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)              # [Batch_size x Seq_length x Hidden_size]
        
        output =  self.dense(context_layer)
        
        return output
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    if model_type == 'LSTM':
        return CaptionModel(hidden_size, embedding_size, vocab.idx)
    elif model_type == 'vanilla_RNN':
        return vanilla_RNN_Model(hidden_size, embedding_size, vocab.idx)
    elif model_type == 'transformer':
        return transformer_Model(config_data, vocab)
#     elif model_type=="attention":
#         selfatten= BertSelfAttention(config_data, vocab)
#         output= selfatten(torch.Tensor(embedding_size))
#         return output
    
