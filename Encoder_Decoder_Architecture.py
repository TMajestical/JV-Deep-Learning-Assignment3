#JV

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

seed = 23
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class BahdanauAttention(nn.Module):
    """
    The class to implement Additive attention aka Bhadanau Attention. As seen in lectures.    
    """
    def __init__(self, hidden_size,D,expected_dim,batch_size):
        super(BahdanauAttention, self).__init__()
        self.U_att = nn.Linear(hidden_size*expected_dim, hidden_size)
        self.W_att = nn.Linear(hidden_size*D, hidden_size)
        self.V_att = nn.Linear(hidden_size, 1)
        self.batch_size = batch_size

    def forward(self, decoder_prev_hidden, encoder_contexts):

        """
        The method that takes decoder hiddenstates and encoder hidden contexts to produce attention weighted context vector.
        Params:
            decoder_prev_hidden : The decoder's hidden state at t-1.
            encoder_contexts : The encoder hidden states from all time steps.
            
        """

        decoder_prev_hidden = decoder_prev_hidden.reshape(self.batch_size,1,-1)
        
        ## Following the same expression as seen in lectures. [Slide 256 in https://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture16.pdf]
        scores = self.V_att(torch.tanh(self.U_att(decoder_prev_hidden.reshape(self.batch_size,1,-1)) + self.W_att(encoder_contexts))).squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        ##compute context of each word in the batch, by considering attention
        context = torch.bmm(weights, encoder_contexts)

        return context, weights
    

class Encoder(nn.Module):
    """
    The class that implements the encoder using Recurrent Units RNN/LSTM/GRU, as needed, by extending the nn.Module class from torch.    
    
    """
    def __init__(self, source_vocab_size,hidden_size,embedding_size,rnn_type = "GRU",padding_idx = None ,dropout=0.1,num_layers = 1,bidirectional = False):
        
        """
        The constructor of the Encoder Class.

        Params:

            source_vocab_size : The vocabulary size of the source language.
            hidden_size : The dimension of the hidden state of the recurrent cell.
            embedding_size : The dimension of the embedding used.
            rnn_type : "GRU"/"LSTM"/"RNN", case INsensitive. Default : "GRU".
            padding_idx : The id corresponding to the token <pad>.
            dropout : Droput probability. Default : 0.1
            num_layers(int) : The number of encoder (recurrence unit) layers. Default : 1
            bidirectional : True/False. If True, encoding is done by parsing input L->R and R->L, hence doubling the hiddenstate size. Default False.

        Return:
            None.
        
        """
        
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.rnn_type = rnn_type

        self.D = 1 ##the number of directions in which the input is viewed.
        if bidirectional:
            self.D = 2

        self.rnn_dropout = 0
        if self.num_layers>1:
            self.rnn_dropout = dropout

        ##create an embedding layer, and ignore padding index
        self.embedding = nn.Embedding(source_vocab_size, self.embedding_size,padding_idx = padding_idx)
        
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(self.embedding_size, hidden_size, batch_first=True,num_layers = num_layers,bidirectional = bidirectional,dropout=self.rnn_dropout)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(self.embedding_size, hidden_size, batch_first=True,num_layers = num_layers,bidirectional = bidirectional,dropout=self.rnn_dropout)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(self.embedding_size, hidden_size, batch_first=True,num_layers = num_layers,bidirectional = bidirectional,dropout=self.rnn_dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input,hidden=None,cell=None):

        """
        The method to perform forward pass of the encoder.

        Params : 
            Input : The encoded batch of input tensors.
            hidden : Default is None. If the unit is LSTM, it is the previous hidden state.
            cell : Default is None. If the unit is LSTM, it is the cell state.

        Returns : 
            output, hidden, cell.
            hidden and cell are current hidden and cell states in case of LSTM and they are None in other cases.
            
        """
        
        input_embedding = self.dropout(self.embedding(input))
        
        if self.rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(input_embedding, (hidden, cell))
        else:
            output, hidden = self.rnn(input_embedding)
            cell = None
        
        return output, hidden, cell
    
    
    
class Decoder(nn.Module):
    """
    The class to implement Decoder in the encoder-decoder architecture using "RNN"/"LSTM"/"GRU".

    While the code is flexible enough to support separate types of recurrent units for encoder and decoder,
    In this assignment, I have chosen to use the same type recurrent unit for both.
    
    """

    def __init__(self, hidden_size,embedding_size,target_vocab_size,rnn_type,batch_size,use_attention = True,padding_idx = None,num_layers = 1,bidirectional = False,dropout=0,device = "cpu"):
        
        """
        The constructor of this class. Perfoms setup necessary for training.

        hidden_size : The dimension of the hidden state of the recurrent cell.
        embedding_size : The dimension of the embedding used.
        target_vocab_size : The vocabulary size of the target language.
        rnn_type : "GRU"/"LSTM"/"RNN", case INsensitive. Default : "GRU".
        batch_size : The batch size used for training. This is needed to resize dimensions in the BahdanauAttention's forward pass.
        use_attention : Boolean variable, default True, indicating to make use of BahdanauAttention.
        padding_idx : The id corresponding to the token <pad>.
        dropout : Droput probability. Default : 0.1
        num_layers(int) : The number of encoder (recurrence unit) layers. Default : 1
        bidirectional : True/False. If True, encoding is done by parsing input L->R and R->L, hence doubling the hiddenstate size. Default False.
        device : The device on which the processing happens. Default : "cpu".
        
        """
        
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.device = device
        self.D = 1 ##the number of directions in which the input is viewed.
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.use_attention = use_attention
        if bidirectional:
            self.D = 2

        ## In h0 (the input to the decoder) first dimension expected is number of directions X number of layers
        self.expected_h0_dim1 = self.D*self.num_layers

        ##create an embedding layer, and ignore padding index
        self.embedding = nn.Embedding(target_vocab_size, self.embedding_size,padding_idx = padding_idx)

        if self.use_attention:
            self.attention = BahdanauAttention(hidden_size,self.D,self.expected_h0_dim1,batch_size)
            recurrent_unit_input_dim = self.embedding_size + self.D*hidden_size

        else:
            recurrent_unit_input_dim = self.embedding_size


        self.rnn_dropout = 0
        if self.num_layers>1:
            self.rnn_dropout = dropout

        #self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True,num_layers = num_layers,bidirectional = bidirectional)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(recurrent_unit_input_dim, hidden_size, batch_first=True,num_layers = num_layers,bidirectional = bidirectional,dropout=self.rnn_dropout)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(recurrent_unit_input_dim, hidden_size, batch_first=True,num_layers = num_layers,bidirectional = bidirectional,dropout=self.rnn_dropout)
        elif self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(recurrent_unit_input_dim, hidden_size, batch_first=True,num_layers = num_layers,bidirectional = bidirectional,dropout=self.rnn_dropout)

        ## Passing the hidden state through a fully conencted layer and then applying softmax
        self.output_layer = nn.Linear(self.hidden_size*self.D, target_vocab_size)

    def forward(self, encoder_hidden_contexts, encoder_last_hidden,encoder_cell,target_tensor,eval_mode = False,teacher_forcing_ratio=0):
        """
        Method that Implements the forward pass of the decoder.

        Params:

            encoder_hidden_contexts : The hidden contexts from all time steps of the encoder.
            encoder_last_hidden : The the last hidden state of the encoder, the context, passed as the first decoder_hidden.
            encoder_cell : It is relevant iff the recurrent unit used is LSTM, all other times it would be 0.
            eval_mode : Boolean variable, if true, it adjusts the dimensions to predict for a single word.
            teacher_forcing_ratio : value in [0,1]. It is essentially the probability, with which true input is fed into the decoder at a time step. Default is 0.

            Returns decoder_outputs,decoder_hidden

        """

        batch_size = encoder_hidden_contexts.size(0)
        if not eval_mode:
            max_word_len = target_tensor.size(1)

        ## eval mode is for looking at a specific word that is predicted to compare with the correct word.
        if eval_mode:
            batch_size = 1
            max_word_len = 30 ## an arbitrary number, larger in expecected sense.

        #the decoder predicts one character at a time, and hence we use a list to store all the predictions.
        decoder_outputs = []
        if self.use_attention:
            attentions = []
        else:
            attentions = None

        ## At the first time step <SOS> token (which has an id 0, is fed as an input to the decoder).
        decoder_input = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        decoder_hidden = encoder_last_hidden ## in the first time step of the decoder, the output of the encoder is the input.
        decoder_cell = encoder_cell ## the cell state, which is initially same as that of encoder, (applies to LSTM unit only)

        for step in range(max_word_len):

            ## eval mode is for looking at a specific word that is predicted to compare with the correct word.
            if eval_mode:
                decoder_input = decoder_input.view(1,-1)

            embedding = self.embedding(decoder_input)


            if decoder_hidden.shape[0] != self.expected_h0_dim1:
                reshaped_hidden = decoder_hidden.repeat(self.expected_h0_dim1,1,1)
            else:
                reshaped_hidden = decoder_hidden


            if self.use_attention:
                ## the attention part.
                decoder_prev_hidden = reshaped_hidden.permute(1, 0, 2)
                context_vector, attention_weights = self.attention(decoder_prev_hidden, encoder_hidden_contexts)
                tmp_input = torch.cat((embedding, context_vector), dim=2)
            else:
                ## introducing non-lineartiy through ReLU activation
                activated_embedding = F.relu(embedding)
                tmp_input = activated_embedding


            if self.rnn_type == "LSTM":
                tmp_output, (decoder_hidden, decoder_cell) = self.rnn(tmp_input, (reshaped_hidden, decoder_cell))
            else:
                tmp_output, decoder_hidden = self.rnn(tmp_input, reshaped_hidden)
                cell = None

            decoder_output = self.output_layer(tmp_output.squeeze(0))

            ## randomly sample a number in (0,1) and if the number is less than the teacher forcing ratio
            ## apply teacher forcing at the current step
            apply_teacher_forcing = random.random() < teacher_forcing_ratio

            if (target_tensor is not None) and (apply_teacher_forcing):

                ## Teacher forcing: Feed the target as the next input
                ## extract the 't'th token from th target string to feed as input at "t"th time step.
                decoder_input = target_tensor[:, step].unsqueeze(1) # Teacher forcing
            else:
                ##greedily pick predictions, i.e pick the character corresponding to the hightest probability
                _,preds = torch.max(decoder_output,dim=2)
                decoder_input = preds.detach()

            decoder_outputs.append(decoder_output)
            if self.use_attention:
                attentions.append(attention_weights)

        ## concatenate the predictions across all the timesteps into a singel tensor
        ## found in literature that log_softmax does better than softmax, hence going with that.
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)

        ## the idea is to have a common API for both attention and normal decoder, achiveing ease of use.
        return decoder_outputs, decoder_hidden, attentions