#JV

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch import optim
import wandb

from Encoder_Decoder_Architecture import *

from tqdm import tqdm

seed = 23
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class MachineTranslator:

    """
    The class that instantiates the encoder-decoder architecture and brings all methods relevant for training, computing accuracy and evaluation here.
    """

    def __init__(self,source_vocab_size,target_vocab_size,hidden_size,embedding_size,rnn_type,batch_size,pad_token_id,dropout,num_layers,bidirectional,use_attention,device):

        """
        The constructor of the class.

        Params:

            source_vocab_size : The vocabulary size of the source language.
            target_vocab_size : The vocabulary size of the target language.
            hidden_size : The dimension of the hidden state of the recurrent cell.
            embedding_size : The dimension of the embedding used.
            rnn_type : "GRU"/"LSTM"/"RNN", case INsensitive. Default : "GRU".
            batch_size : The batch size used for training. This is needed to resize dimensions in the BahdanauAttention's forward pass.
            pad_token_id : The id corresponding to the token <pad>.
            dropout : Droput probability. Encoder and Decoder by default use a dropout of 0.1, unless specified otherwise.
            num_layers(int) : The number of encoder (recurrence unit) layers. Default : 1
            bidirectional : True/False. If True, encoding is done by parsing input L->R and R->L, hence doubling the hiddenstate size. Default False.
            use_attention : Boolean variable, default True, indicating to make use of BahdanauAttention.

            Note : hidden_size,embedding_size,dopout, num_layers,bidirectional, rnn_type. These parameters are consistent across the encoder and decoder.
                    However, the code supports use of different values.
        
        Returns:

            None.

        
        """
        self.device = device
        
        self.encoder = Encoder(source_vocab_size = source_vocab_size, hidden_size = hidden_size,embedding_size=embedding_size,rnn_type = rnn_type,padding_idx=pad_token_id,num_layers=num_layers,bidirectional=bidirectional,dropout=dropout).to(self.device)
    
        self.decoder = Decoder(hidden_size = hidden_size,embedding_size=embedding_size, target_vocab_size = target_vocab_size,batch_size = batch_size,rnn_type = rnn_type,use_attention = use_attention, padding_idx = pad_token_id,num_layers = num_layers,bidirectional = bidirectional,dropout=dropout,device=self.device).to(self.device)

        
    def train_epoch(self,train_loader, encoder, decoder, encoder_optim,decoder_optim, loss_criterion,teacher_forcing_ratio,ignore_padding=True,device='cpu'):

        """
        Method to train the encoder-decoder model for 1 epoch.

        Params:

            train_loader : The dataloader object (which wraps around WordDataset object of Core_Utils) corresponding to the traindata.
            encoder : The encoder model object.
            decoder : The decoder model object.
            encoder_optim : A torch optim object, corresponding to the optimizer of encoder.
            decoder_optim : A torch optim object, corresponding to the optimizer of decoder.
            loss_criterion : The loss criterion
            teacher_forcing_ratio : The teacher forcing ratio to be used.
            ignore_padding : True, by default.
            device : CPU by default.

        Returns:
            Loss and accuracy of the current epoch.
        
        """

        tot_correct_word_preds = 0
        tot_words = 0
        epoch_loss = 0

        for data in tqdm(train_loader):

            input_tensor, target_tensor,_,_ = data

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()

            batch_size = data[0].shape[0]

            if encoder.rnn_type == "LSTM":
                encoder_hidden = torch.zeros(encoder.num_layers*encoder.D, batch_size, encoder.hidden_size, device=device)
                encoder_cell = torch.zeros(encoder.num_layers*encoder.D, batch_size, encoder.hidden_size, device=device)
            else:
                encoder_hidden = None
                encoder_cell = None            

            encoder_hidden_contexts, encoder_last_hidden, encoder_cell = encoder(input_tensor,encoder_hidden,encoder_cell)
            
            decoder_outputs, _, _ = decoder(encoder_hidden_contexts, encoder_last_hidden,encoder_cell, target_tensor=target_tensor,teacher_forcing_ratio = teacher_forcing_ratio)

            multi_step_preds = torch.argmax(decoder_outputs,dim=2)
            multi_step_pred_correctness = (multi_step_preds ==  target_tensor)
            num_words = multi_step_preds.shape[0]
            
            if ignore_padding: ## if padding has to be ignored.

                ## for each word, based on the padding token ID, find the first occurance of the padding token, marking the begining of padding.
                ## argmax is not supported for bool on cuda, hence casting it to long.
                padding_start = torch.argmax((target_tensor == train_loader.dataset.pad_token_id).to(torch.long),dim=1).to(device)
                ## Creating a mask with 1's in each position of a padding token
                mask = (torch.arange(target_tensor.size(1)).unsqueeze(0).to(device) >= padding_start.unsqueeze(1))
                
                ##doing a logical OR with the mask makes sure that the padding tokens do not affect the correctness of the word
                tot_correct_word_preds += (torch.all(torch.logical_or(multi_step_pred_correctness,mask),dim=1).int().sum()).item()
                tot_words += num_words

            loss = loss_criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            encoder_optim.step()
            decoder_optim.step()

            epoch_loss += loss.item()

        epoch_loss = round(epoch_loss / len(train_loader),4)
        epoch_accuracy = round(tot_correct_word_preds*100/tot_words,2)

        return epoch_loss,epoch_accuracy
    
    def train(self,train_loader,valid_loader, encoder, decoder, epochs,padding_idx,optimiser = "adam",loss="crossentropy",weight_decay=0, lr=0.001,teacher_forcing_ratio = 0,device='cpu',wandb_logging = False):

        """
        The method to train the encoder-decoder model. Makes use of other methods like train_epoch, compute_accuracy to train and return the accuracy.

        train_loader : The dataloader object (which wraps around WordDataset object of Core_Utils) corresponding to the traindata.
        valid_loader : The dataloader object (which wraps around WordDataset object of Core_Utils) corresponding to the Validaiton data.
        encoder : The encoder model object.
        decoder : The decoder model object. 
        epochs : Number of epochs of training.
        padding_idx : The index corresponding to the <pad> token.
        optimiser : The optimiser used for training, "adam"/"nadam"/"rmsprop", default : "adam". (Case sensitive)
        loss : The loss function, only "crossentropy" is supported.
        weight_decay : L2, regularization of encoder and decoder model weights.
        lr : The learning rate, default is 0.001
        teacher_forcing_ratio : Teacher forcing ratio, default is 0.
        device : Default is CPU.
        wandb_logging : Default is False.
        
        """
    
        ## specify the optimiser
        if optimiser.lower() == "adam":
            encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr,weight_decay=weight_decay)
            decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr,weight_decay=weight_decay)

        elif optimiser.lower() == "nadam":
            encoder_optimizer = optim.NAdam(encoder.parameters(), lr=lr,weight_decay=weight_decay)
            decoder_optimizer = optim.NAdam(decoder.parameters(), lr=lr,weight_decay=weight_decay)

        elif optimiser.lower() == "rmsprop":
            encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=lr,weight_decay=weight_decay)
            decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=lr,weight_decay=weight_decay)
            
        ## Specify the loss criteria
        if loss.lower() == "crossentropy":
            loss_criterion = nn.CrossEntropyLoss(ignore_index = padding_idx).to(device)

        lp = train_loader.dataset.lp
        
        for epoch in tqdm(range(epochs)):

            ## Train for 1 epoch.
            train_loss,train_accuracy = self.train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_criterion,teacher_forcing_ratio,device=device)

            ## compute validation accuracy.
            val_loss,_,val_accuracy = self.compute_accuracy(valid_loader,encoder,decoder,loss_criterion,ignore_padding=True,device=device)

            print(f"Epoch {epoch+1}\t Train Loss : {train_loss}\t Train Acc : {train_accuracy}% \t Val Loss : {val_loss}\t Val Acc : {val_accuracy}%")
            if wandb_logging:
                wandb.log({'epoch': epoch+1,'train loss': train_loss, 'train accuracy': train_accuracy, 'Validation loss': val_loss, 'Validation accuracy': val_accuracy})

    def compute_accuracy(self,dataloader,encoder,decoder,criterion,ignore_padding = True,device='cpu',save_results=False,filename=""):

        """
        Method to compute the accuracy using the model (encoder-decoder) using dataloader.

        This method returns word and character level accuracy.

            Word Level Accuracy : Accuracy is computed at the word level and a word is right iff every character is predicted correctly.
            Char Level Accuracy : Accuracy is computed by comparing each predicted character wrt the correct char.

        Params:

            dataloader : The train/test/valid dataloader.
            encoder : The encoder 
            decoder : The decoder
            padding_token_id : The id of the padding token.
            ignore_padding : If True, then in word level accuracy, the padding characters are ignored in computing the word level accuracy.
                            char level accuracy, the padding characters are not considered at all.

                            If false, padding is considered to be a part of the word (for word level accuracy) and the entire word is checked.
            device : Default CPU.
            save_results : Default is False. If true, the results of predictions are saved to a CSV, named by the next param.
            filename : The name of the file, if results have to be saved.

        """

        char_lvl_accuracy = 0
        word_level_accuracy = 0

        tot_chars = 0
        tot_words = 0

        tot_correct_char_preds = 0
        tot_correct_word_preds = 0

        loss = 0

        if save_results:
            rows = []

        #criterion = loss_criterion.to(device)

        with torch.no_grad():

            train = 0

            if encoder.training and decoder.training: ## reset the the model back to train mode
                train = 1

            encoder.eval()
            decoder.eval()

            for data in dataloader:
                
                input_tensor, target_tensor,_,target_max_len = data

                batch_size = data[0].shape[0]
        
                if encoder.rnn_type == "LSTM":
                    encoder_hidden = torch.zeros(encoder.num_layers*encoder.D, batch_size, encoder.hidden_size, device=device)
                    encoder_cell = torch.zeros(encoder.num_layers*encoder.D, batch_size, encoder.hidden_size, device=device)
                else:
                    encoder_hidden = None
                    encoder_cell = None
                
        
                encoder_hidden_contexts, encoder_last_hidden, encoder_cell = encoder(input_tensor,encoder_hidden,encoder_cell)
                ## even though we are passing target tensor, the teacher forcing ratio is 0, so no teacher forcing
                decoder_outputs, _, attentions = decoder(encoder_hidden_contexts, encoder_last_hidden,encoder_cell, target_tensor = target_tensor,teacher_forcing_ratio = 0)
                loss += criterion(decoder_outputs.view(-1, decoder_outputs.size(-1)), target_tensor.view(-1)).item()
        
                ## For a batch, for each character find the most probable output word.
                multi_step_preds = torch.argmax(decoder_outputs,dim=2)
                multi_step_pred_correctness = (multi_step_preds ==  target_tensor)
                num_chars = multi_step_preds.numel() ##find the total number of characters in the current batch
                num_words = multi_step_preds.shape[0] ##find the total number of words in the current batch.
        
                if ignore_padding: ## if padding has to be ignored.
        
                    ## for each word, based on the padding token ID, find the first occurance of the padding token, marking the begining of padding.
                    
                    ## argmax is not supported for bool on cuda, hence casting it to long.
                    padding_start = torch.argmax((target_tensor == dataloader.dataset.pad_token_id).to(torch.long),dim=1).to(device)
                    ## Creating a mask with 1's in each position of a padding token
                    mask = (torch.arange(target_tensor.size(1)).unsqueeze(0).to(device) >= padding_start.unsqueeze(1))
                    #print(mask)
                    
                    ##doing a logical OR with the mask makes sure that the padding tokens do not affect the correctness of the word
                    tot_correct_word_preds += (torch.all(torch.logical_or(multi_step_pred_correctness,mask),dim=1).int().sum()).item()
                    tot_words += num_words
        
                    ##creating a complement of the mask so to mark padding tokens as irrelevant
                    complement_mask = (1-mask.int()).bool()
                    num_pad_chars = mask.int().sum().item()
                    ##counting number of non_pad_chars to compute accuracy.
                    num_non_pad_chars = num_chars - num_pad_chars
        
                    tot_correct_char_preds += (torch.logical_and(multi_step_pred_correctness,complement_mask).int().sum()).item()
                    tot_chars += num_non_pad_chars

                    if save_results:
                        word_preds_correctness = torch.all(torch.logical_or(multi_step_pred_correctness,mask),dim=1).int()
                        for i in range(multi_step_preds.shape[0]):
                            rows.append([dataloader.dataset.lp.decode_word(input_tensor[i].cpu().numpy(),lang_id=0),dataloader.dataset.lp.decode_word(target_tensor[i].cpu().numpy(),lang_id=1),dataloader.dataset.lp.decode_word(multi_step_preds[i].cpu().numpy(),lang_id=1),word_preds_correctness[i].cpu().item()])
                    
            
                else: ##otherwise.
        
                    tot_correct_word_preds += (torch.all(multi_step_pred_correctness,dim=1).int().sum()).item()
                    tot_words += num_words
                    
                    tot_correct_char_preds += (multi_step_pred_correctness.int().sum()).item()
                    tot_chars += num_chars

            char_lvl_accuracy = round(tot_correct_char_preds*100/tot_chars,2)
            word_lvl_accuracy = round(tot_correct_word_preds*100/tot_words,2)

            loss /= dataloader.dataset.data.shape[0]

            if save_results:
                df = pd.DataFrame(data=rows, columns=["Source Word","Target Word","Predicted Word","Is Prediction Correct"])
                df.to_csv(filename+".csv",index=False)

            if train:

                encoder.train()
                decoder.train()
        
            return round(loss,4),char_lvl_accuracy,word_lvl_accuracy