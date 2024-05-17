#JV

import numpy as np

import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence
import random
from functools import partial

import numpy as np

from tqdm.notebook import tqdm

#!python3 -m pip install wandb
import wandb

import argparse

from Core_Utils import *

from Encoder_Decoder_Architecture import *

from Machine_Translator import *

seed = 23

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def setup_and_start_expt(config,wandb_log = False):

    """
    Method to run experiments as per the specified config and parameters.

    config : A dictionary of paramter-values.
    wandb_log : Default False, if true, the accuracies would be logged.

    Returns:
        Trained MachineTranslator model.
    """
    
    batch_size = config['batch_size']
    target_lang = "tel"

    base_dir = "aksharantar_sampled/"

    if config['device'] == None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config['device'])

    use_meta_tokens = True
    
    lang_dir = base_dir + target_lang + "/"
    
    ##creating train loader
    train_lp = LanguageProcessor(language_directory=lang_dir,target_lang_name=target_lang,mode="train",meta_tokens=use_meta_tokens)
    ## the ids of these tokens are the same in the source and target language
    pad_token_id = train_lp.source_char2id['<pad>']

    collate_fn_ptr = partial(collate_fn,pad_token_id=pad_token_id,device=device)
    
    train_dataset = WordDataset(train_lp,device=device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,collate_fn=collate_fn_ptr, shuffle=True)
    
    ## creating test loader
    test_lp = LanguageProcessor(language_directory=lang_dir,target_lang_name=target_lang,mode="test",meta_tokens=use_meta_tokens)
    
    ## to make sure that the same vocabulary and dictionaries are used everywhere
    test_lp.source_char2id = train_lp.source_char2id
    test_lp.source_id2char = train_lp.source_id2char
    test_lp.target_char2id = train_lp.target_char2id
    test_lp.target_id2char = train_lp.target_id2char
    
    test_dataset = WordDataset(test_lp,device=device)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,collate_fn=collate_fn_ptr, shuffle=True)
    
    ## creating validation loader
    valid_lp = LanguageProcessor(language_directory=lang_dir,target_lang_name=target_lang,mode="valid",meta_tokens=use_meta_tokens)
    valid_lp.source_char2id = train_lp.source_char2id
    valid_lp.source_id2char = train_lp.source_id2char
    valid_lp.target_char2id = train_lp.target_char2id
    valid_lp.target_id2char = train_lp.target_id2char
    
    valid_dataset = WordDataset(valid_lp,device=device)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,collate_fn=collate_fn_ptr, shuffle=True)
    
    ##in principle these are all fixed across train/test/valid data
     
    source_vocab_size = len(list(train_lp.source_char2id.keys()))
    target_vocab_size = len(list(train_lp.target_char2id.keys()))
    
    hidden_size = config['hidden_size']
    embedding_size = hidden_size
    
    epochs = config['epochs']
    
    optimiser = config['optimiser']
    
    weight_decay = config['weight_decay']
    
    lr = config['lr']
    
    num_layers = config['num_layers']
    
    ## Allowed Values : "GRU"/"RNN"/"LSTM" (not case sensitive)
    rnn_type = config['rnn_type'].upper()
    
    bidirectional = config['bidirectional']
    teacher_forcing_ratio = config['teacher_forcing_ratio']

    dropout=config['dropout']

    use_attention = config['use_attention']
    
    model = MachineTranslator(source_vocab_size,target_vocab_size,hidden_size,embedding_size,rnn_type,batch_size,pad_token_id,dropout,num_layers,bidirectional,use_attention,device)

    model.train(train_loader,valid_loader,model.encoder, model.decoder, epochs,padding_idx = pad_token_id,optimiser = optimiser,weight_decay=weight_decay, lr=lr,teacher_forcing_ratio = teacher_forcing_ratio,device=device,wandb_logging = wandb_log)
    
    loss_criterion = nn.CrossEntropyLoss(ignore_index = pad_token_id).to(device)
    test_loss,_,test_accuracy = model.compute_accuracy(test_loader,model.encoder,model.decoder,loss_criterion,ignore_padding=True,device=device,save_results=True,filename = "predictions_attention")
    print(f"Testing Loss : {test_loss}\tTesting Accuracy : {test_accuracy}%")

    return model

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=64,help="Batch size used to train neural network.")

parser.add_argument("-bid", "--bidirectional", type=bool,default=True,choices=[True,False],help="If True, input would be seen in both directions.")

parser.add_argument("-dpt", "--dropout", type=float, default=0.4,help="The dropout probability.")

parser.add_argument("-es", "--embedding_size", type=int, default=256,help="The input embedding dimension.")

parser.add_argument("-e", "--epochs", type=int, default=15,help="Number of epochs to train.")

parser.add_argument("-hs", "--hidden_size", type=int, default=512,help="The dimension of the hidden state.")

parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4,help="Learning rate used to optimize model parameters.")

parser.add_argument("-nl", "--num_layers", type=int, default=4,help="Number of Recurrence Layers.")

parser.add_argument("-o", "--optimizer", type=str, default="nadam",choices=["rmsprop", "adam", "nadam"],help="Optimizer used to minimize the loss.")

parser.add_argument("-rt", "--rnn_type", type=str, default="lstm",choices=["lstm", "gru", "rnn"],help="The type of recurrent cell to be used.")

parser.add_argument("-tf", "--teacher_forcing", type=float, default=0.4,help="The Teacher Forcing Ratio.")

parser.add_argument("-w_d", "--weight_decay", type=float, default=1e-5,help="Weight decay used by optimizers.")

parser.add_argument("-d", "--device", type=str, default=None,help="The device on which the training happens.")

args = parser.parse_args()

config = {

        'batch_size':args.batch_size,

        'bidirectional' :  args.bidirectional,

        'dropout' : args.dropout,

        'embedding_size':args.embedding_size,

        'epochs' : args.epochs,

        'hidden_size':args.hidden_size,

        'lr':  args.learning_rate,

        'num_layers' : args.num_layers,

        'optimiser': args.optimizer,

        'rnn_type' : args.rnn_type,

        'teacher_forcing_ratio' : args.teacher_forcing,

        'weight_decay': args.weight_decay,

        'device' : args.device,

        'use_attention' : False
}

model = setup_and_start_expt(config)