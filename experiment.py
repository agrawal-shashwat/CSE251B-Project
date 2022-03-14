################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import math
from vqaTools.vqa import VQA
import torch.nn.functional as F
import string
import nltk

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__vqa_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_valLoss = math.inf
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = torch.nn.CrossEntropyLoss(ignore_index = 0)
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr = config_data['experiment']['learning_rate'])

        self.__init_model()
        # Load Experiment Data if available
        self.__load_experiment()
        

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        print("Load Experiment")
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)
            self.__best_valLoss = min(self.__val_losses) if len(self.__training_losses) > 0 else math.inf  
            model_Path = os.path.join(self.__experiment_dir, 'latest_model.pt') 
            if os.path.exists(model_Path):
                self.__load_model(model_Path)
            else:
                self.__model.encoder.load_state_dict(torch.load('./premodels/encoder-best.ckpt')) # put checkpoint name
                self.__model.decoder.load_state_dict(torch.load('./premodels/decoder-best.ckpt'))
        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        print("Run Experiment")
        start_epoch = self.__current_epoch
        print("Start epoch",start_epoch)
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        print("Train Experiment")
        self.__model.train()
        training_loss = 0
            self.__optimizer.zero_grad()

            # Transfer Input & Label to the model's device
            inputs = images.cuda()
            labels = captions.cuda()
<<<<<<< Updated upstream
            outputs, _ = self.__model(inputs, labels)
<<<<<<< HEAD
            loss = self.__criterion(outputs, labels[:, 1:])
=======
            answers = answers.cuda()
            outputs, _ = self.__model(inputs, labels,answers)
            loss = self.__criterion(outputs.view(-1,vocab_size), labels[:, 1:].reshape(-1))
>>>>>>> Stashed changes
=======
>>>>>>> d14b97ee73aa63071e3ea080e3d7011d8d9ad1a4
            print("train iteration ",i,loss.item())
            training_loss += loss.item()

            # Backpropagate & update weights
            loss.backward()
            self.__optimizer.step()

        return training_loss/len(self.__train_loader)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0
        vocab_size = self.__vocab.idx
        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                inputs = images.cuda()
                labels = captions.cuda()
                answers = answers.cuda()
                print("validation iteration ",i)
<<<<<<< Updated upstream
                outputs, _ = self.__model(inputs, labels)
<<<<<<< HEAD
                loss = self.__criterion(outputs, labels[:, 1:])
=======
                outputs, _ = self.__model(inputs, labels,answers)
                loss = self.__criterion(outputs.view(-1,vocab_size), labels[:, 1:].reshape(-1))
>>>>>>> Stashed changes
=======
>>>>>>> d14b97ee73aa63071e3ea080e3d7011d8d9ad1a4
                val_loss += loss.item()
                
            val_loss /= len(self.__val_loader)
                
            if val_loss < self.__best_valLoss:
                self.__best_valLoss = val_loss
                model_dict = self.__model.state_dict()
                state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
                self.__best_model = state_dict
                self.__save_model(True)

        return val_loss

    def clean_sentence(self,output):
        config_data = read_file_in_dir('./', 'default.json')
        max_Caption_Length = config_data['generation']['max_length']
        cleaned_list = []
        for index in output:
            if  (index in [0,1,2,3]) : #ignore padding
                continue
            cleaned_list.append(word.lower())                                                             
        tokens = nltk.tokenize.word_tokenize(' '.join(cleaned_list))
        return tokens

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        
        config_data = read_file_in_dir('./', 'default.json')
        test_loss = 0
        bleu1Val = 0
        bleu4Val = 0
        meteorVal = 0
        vocab_size = self.__vocab.idx
        model_Path = os.path.join(self.__experiment_dir, 'latest_model_best.pt') 
        if os.path.exists(model_Path):
            self.__load_model(True)
        
        self.__model.eval()
        
        vqa = self.__vqa_test
        batch_size = config_data['dataset']['batch_size']
        temperature = config_data['generation']['temperature']
        max_Caption_Length = config_data['generation']['max_length']
        reference_all, cleaned_all = [], []
        
        with torch.no_grad():
                inputs = images.cuda()
                labels = captions.cuda()
                answers = answers.cuda()
                print("test iteration ",iter)
                
                # Produce teacher output for loss
<<<<<<< Updated upstream
                outputs, _ = self.__model(inputs, labels)
<<<<<<< HEAD
                loss = self.__criterion(outputs, labels[:, 1:])
=======
                print("Test start")
                outputs, _ = self.__model(inputs, labels,answers)
                loss = self.__criterion(outputs.view(-1,vocab_size), labels[:, 1:].reshape(-1))
>>>>>>> Stashed changes
=======
                loss = self.__criterion(outputs.view(-1,vocab_size), labels[:, 1:].reshape(-1))
>>>>>>> d14b97ee73aa63071e3ea080e3d7011d8d9ad1a4
                test_loss += loss.item()
                
                # Produce non-teacher outputs for Bleu
                predicted = []
                for j in range(labels.shape[1] - 1):
                    labelOutputs, state = self.__model(inputs, labels[:, 1:], state = 'init' if j == 0 else state)
                    labelOutputs = labelOutputs.permute(0, 2, 1)
                    labelOutputs = F.softmax(labelOutputs / temperature, dim = -1)
                    labelOutputs = torch.multinomial(labelOutputs.squeeze(1).data, 1)
                    predicted.append(labelOutputs)
                    inputs = labelOutputs.clone()
                    inputs[inputs == 2] = 0 # If output is <end>, convert input to <pad>  
                predicted = torch.stack(predicted, dim = 1)
#                 predicted = []
#                 for j in range(labels.shape[1] - 1):
#                     labelOutputs, state = self.__model(inputs, labels[:, 1:], state = 'init' if j == 0 else state)
#                     labelOutputs = labelOutputs.permute(0, 2, 1)
#                     labelOutputs = F.softmax(labelOutputs / temperature, dim = -1)
#                     labelOutputs = torch.multinomial(labelOutputs.squeeze(1).data, 1)
#                     predicted.append(labelOutputs)
#                     inputs = labelOutputs.clone()
#                     inputs[inputs == 2] = 0 # If output is <end>, convert input to <pad>  
#                 predicted = torch.stack(predicted, dim = 1)
                
                # Compute Bleu                       
                for index, ques_id in enumerate(ques_ids):
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
                    features = self.__model.encoder(inputs[index:index+1])
                    caps,alphas = self.__model.decoder.generate_caption(features,config_data,answers[index:index+1],self.__vocab)                   
>>>>>>> Stashed changes
=======
                    features = self.__model.encoder(inputs[index:index+1])
                    caps,alphas = self.__model.decoder.generate_caption(features,config_data, self.__vocab)                   
>>>>>>> d14b97ee73aa63071e3ea080e3d7011d8d9ad1a4
                    referenceCaptions = []
                    actualCaptions = []
                    refImage = images[index]
                    
                    for x in [vqa.qqa[ques_id]]:
                        caption = x['question'].lower()
                        caption = caption.translate(str.maketrans('', '', string.punctuation))
                        tokens = nltk.tokenize.word_tokenize(caption)
                        referenceCaptions.append(tokens)  
                        actualCaptions.append(' '.join(tokens))
                    
                    # cleanedSentence = self.clean_sentence(predicted[index]) 
                    # if index%10 == 0:
                    #     print("Predicted Sentence ", cleanedSentence)
                    #     print("Reference Captions ", actualCaptions)
                    reference_all.append(referenceCaptions)
        
        lengthOfSet = len(self.__test_loader)
        bleu1Val = bleu1(reference_all, cleaned_all)
        bleu4Val = bleu4(reference_all, cleaned_all)
                                                                                                test_loss/lengthOfSet,
                                                                                               bleu1Val,
                                                                                              bleu4Val,
                                                                                               meteorVal)
        self.__log(result_str, 'epoch.log')


            
    def __save_model(self, isBestModel = False):
        
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')              
        
        if isBestModel == True:         
            root_model_path = os.path.join(self.__experiment_dir, 'latest_model_best.pt')
        
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)   


    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
        
    def __load_model(self, isBestModel = False):
        model_Path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        
        if isBestModel == True:
            model_Path = os.path.join(self.__experiment_dir, 'latest_model_best.pt') 
        state_dict = torch.load(model_Path)
        self.__model.load_state_dict(state_dict['model'])
        
        self.__optimizer.load_state_dict(state_dict['optimizer'])
