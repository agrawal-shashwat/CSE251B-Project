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
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        
        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)
        
        
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
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)#cat both image and the question
        outputs = self.decoder(features, captions)
        return outputs
    
    
########################################################



# class ImgEncoder(nn.Module):

#     def __init__(self, embed_size):
#         """(1) Load the pretrained model as you want.
#                cf) one needs to check structure of model using 'print(model)'
#                    to remove last fc layer from the model.
#            (2) Replace final fc layer (score values from the ImageNet)
#                with new fc layer (image feature).
#            (3) Normalize feature vector.
#         """
#         super(ImgEncoder, self).__init__()
#         model = models.vgg19(pretrained=True)
#         in_features = model.classifier[-1].in_features  # input size of feature vector
#         model.classifier = nn.Sequential(
#             *list(model.classifier.children())[:-1])    # remove last fc layer

#         self.model = model                              # loaded model without last fc layer
#         self.fc = nn.Linear(in_features, embed_size)    # feature vector of image

#     def forward(self, image):
#         """Extract feature vector from image vector.
#         """
#         with torch.no_grad():
#             img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
#         img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

#         l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
#         img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

#         return img_feature


# class QstEncoder(nn.Module):

#     def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

#         super(QstEncoder, self).__init__()
#         self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
#         self.tanh = nn.Tanh()
#         self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
#         self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

#     def forward(self, question):

#         qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
#         qst_vec = self.tanh(qst_vec)
#         qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
#         _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
#         qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
#         qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
#         qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
#         qst_feature = self.tanh(qst_feature)
#         qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

#         return qst_feature



# class ImgAttentionEncoder(nn.Module):

#     def __init__(self, embed_size):
#         """(1) Load the pretrained model as you want.
#                cf) one needs to check structure of model using 'print(model)'
#                    to remove last fc layer from the model.
#            (2) Replace final fc layer (score values from the ImageNet)
#                with new fc layer (image feature).
#            (3) Normalize feature vector.
#         """
#         super(ImgAttentionEncoder, self).__init__()
#         vggnet_feat = models.vgg19(pretrained=True).features
#         modules = list(vggnet_feat.children())[:-2]
#         self.cnn = nn.Sequential(*modules)
#         self.fc = nn.Sequential(nn.Linear(self.cnn[-3].out_channels, embed_size),
#                                 nn.Tanh())     # feature vector of image

#     def forward(self, image):
#         """Extract feature vector from image vector.
#     #     """
#         with torch.no_grad():
#             img_feature = self.cnn(image)                           # [batch_size, vgg16(19)_fc=4096]
#         img_feature = img_feature.view(-1, 512, 196).transpose(1,2) # [batch_size, 196, 512]
#         img_feature = self.fc(img_feature)                          # [batch_size, 196, embed_size]

#         return img_feature


# class Attention(nn.Module):
#     def __init__(self, num_channels, embed_size, dropout=True):
#         """Stacked attention Module
#         """
#         super(Attention, self).__init__()
#         self.ff_image = nn.Linear(embed_size, num_channels)
#         self.ff_questions = nn.Linear(embed_size, num_channels)
#         self.dropout = nn.Dropout(p=0.5)
#         self.ff_attention = nn.Linear(num_channels, 1)

#     def forward(self, vi, vq):
#         """Extract feature vector from image vector.
#         """
#         hi = self.ff_image(vi)
#         hq = self.ff_questions(vq).unsqueeze(dim=1)
#         ha = torch.tanh(hi+hq)
#         if self.dropout:
#             ha = self.dropout(ha)
#         ha = self.ff_attention(ha)
#         pi = torch.softmax(ha, dim=1)
#         self.pi = pi
#         vi_attended = (pi * vi).sum(dim=1)
#         u = vi_attended + vq
#         return u

# class SANModel(nn.Module):
#     # num_attention_layer and num_mlp_layer not implemented yet
#     def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size): 
#         super(SANModel, self).__init__()
#         self.num_attention_layer = 2
#         self.num_mlp_layer = 1
#         self.img_encoder = ImgAttentionEncoder(embed_size)
#         self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
#         self.san = nn.ModuleList([Attention(512, embed_size)]*self.num_attention_layer)
#         self.tanh = nn.Tanh()
#         self.mlp = nn.Sequential(nn.Dropout(p=0.5),
#                             nn.Linear(embed_size, ans_vocab_size))
#         self.attn_features = []  ## attention features

#     def forward(self, img, qst):

#         img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
#         qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
#         vi = img_feature
#         u = qst_feature
#         for attn_layer in self.san:
#             u = attn_layer(vi, u)
# #             self.attn_features.append(attn_layer.pi)
            
#         combined_feature = self.mlp(u)
#         return combined_feature

# Build and return the model here based on the configuration.
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
