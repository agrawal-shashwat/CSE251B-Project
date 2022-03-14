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
<<<<<<< Updated upstream
        
class vanilla_RNN_Model(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size):
        super(vanilla_RNN_Model, self).__init__()
        
<<<<<<< Updated upstream
        # Load pretrained ResNet50 and remove last FC layer
        self.resnet = nn.Sequential(*list(torchvision.models.resnet50(pretrained = True).children())[:-1])
        for param in self.resnet.parameters():
            param.requires_grad = False
=======
        
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
<<<<<<< Updated upstream
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
=======
            embeds = self.embedding(word)      

        
        captions = []
        
        for i in range(max_len):
            alpha,context = self.attention(features, h)
            
            
            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
           
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            #output = output.view(batch_size,-1)
        
            
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
    
    
    def init_hidden_state(self, encoder_out,answ_embeds , dim_encoder_mean = 1):
#         mean_encoder_out = encoder_out
#         if dim_encoder_mean == 1:
        mean_encoder_out = encoder_out.mean(dim=1)
        
        ans_embeds = torch.sum(answ_embeds,dim=1)
    
        mean_encoder_out = torch.cat((ans_embeds, mean_encoder_out), dim=1)
 
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
        ans_embed_size = config_data['model']['answer_embed_size']
     
        drop_prob=0.3
        self.encoder = EncoderCNN()#replace with image and question encoder
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            answer_embed_dim = ans_embed_size
        )
        
    def forward(self, images, captions,answers=None):
        features = self.encoder(images)#cat both image and the question
        outputs = self.decoder(features, captions,answers)
        return outputs
    
>>>>>>> Stashed changes
    
# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    if model_type == 'LSTM':
        return CaptionModel(hidden_size, embedding_size, vocab.idx)
    elif model_type == 'vanilla_RNN':
        return vanilla_RNN_Model(hidden_size, embedding_size, vocab.idx)
