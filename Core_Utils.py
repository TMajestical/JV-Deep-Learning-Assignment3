#JV

import numpy as np
import pandas as pd
import gc

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class LanguageProcessor:

    def __init__(self,language_directory,target_lang_name,mode="train",meta_tokens=True):

        """
        Default Constructor for this class.

        Params:

            language_directory : ex : "aksharantar_sampled/tel/"
            mode : "train" or "test" or "valid", accordingly the appropriate dataset is read.
            meta_tokens : If true creates the first three tokens of the dictionary as <start>,<end>,<pad>.
            
        """

        self.meta_tokens = meta_tokens ## boolean variable, if 1, then <start>,<end> and <pad> tokens are cosidered in the vocab.
        self.language_directory = language_directory
        self.target_lang_name = target_lang_name
        self.mode = mode ## accordingly helps to read and generate the appropriate dataset.
    
        self.source_lang = 0
        self.target_lang = 1

        self.source_char2id,self.source_id2char = self.build_char_vocab(self.source_lang)
        self.target_char2id,self.target_id2char = self.build_char_vocab(self.target_lang)


    def build_char_vocab(self,lang_id,max_len=None):

        """
        Method to create a vocabulary of characters in language corresponding to lang_id.

        lang_id : 0/1 indicating the source and target language respectively.
        max_len : Depricated!

        Returns:
            None.
        """

        df = pd.read_csv(self.language_directory+self.target_lang_name+"_"+self.mode+".csv",header=None)

        self.data = df.to_numpy()

        lang_chars = []
        lang_words = df[lang_id].to_numpy()
    
        for word in lang_words:
            lang_chars += list(word)
    
        unique_lang_chars =  sorted(list(set(lang_chars)))
        
        if self.meta_tokens:
            char2id_dict = {'<start>':0,'<end>':1,'<pad>': 2}
            id2char_dict = {0:'<start>',1:'<end>',2:'<pad>'}

            ## These ids would still hold irrespective of the target language
            self.start_token_id = char2id_dict['<start>']
            self.end_token_id = char2id_dict['<end>']
            self.pad_token_id = char2id_dict['<pad>']
            
        else:
            char2id_dict = {}
            id2char_dict = {}

        start = len(char2id_dict) ##Key of each language character starts based on meta tokens are used or not.
    
        for i in range(len(unique_lang_chars)):
            char2id_dict[unique_lang_chars[i]] = i+start
            id2char_dict[i+start] = unique_lang_chars[i]
    
        del df
        del lang_chars
        del unique_lang_chars

        gc.collect()
    
        return char2id_dict,id2char_dict

    def encode_word(self,word,lang_id,padding=False,append_eos = False):

        """
        Method to encode characters of a given word.

        Params:

            word: The word to be encoded.
            lang_id : 0/1 for source/target lang.
            padding [Default - False] : If true, the word encoding would be padded upto max len.
            append_eos [Default - False] : Appends <end> token at the end of every word.

        Returns:

            Encoded Word (np array of character id's)
        
        """

        if lang_id == self.source_lang:
            char2id_dict = self.source_char2id
            
        else:
            char2id_dict = self.target_char2id
        
        word_encoding = []
        
        for i in word.lower():
            word_encoding.append(char2id_dict[i])

        if append_eos:
            word_encoding.append(char2id_dict['<end>'])

        return np.array(word_encoding)

    def decode_word(self,code_word,lang_id):

        """
        Method to decode an encoded word.

        Params:

            code_word : The encoded word.
            lang_id : 0/1 for source/target lang.
        """
    
        word = []

        if lang_id == self.source_lang:
            id2char_dict = self.source_id2char
            char2id_dict = self.source_char2id
            
        else:
            id2char_dict = self.target_id2char
            char2id_dict = self.target_char2id

        start_idx = 0
        
        for i in code_word[start_idx:]:
            ## if we reached <end>, then stop decoding
            if self.meta_tokens and i == char2id_dict['<end>'] or i == char2id_dict['<pad>']:
                break
            
            word.append(id2char_dict[i])
            
        return np.array(word)
            
class WordDataset(Dataset):

    """
    Class that inherits and overrides the methods of Dataset class. This helps in creating a data loader.
    """
    
    def __init__(self, language_processor,append_eos=True,device='cpu'):

        """
        The constructor of the class.

        Params:

            language_processor : An object of the class LanguageProcessor, helps to encode and decode words into string of integers.
            append_eos: A boolean param, if True, an <eos> tag is added at the end of the encoded input.
            device : The device on which the processing happens, default : CPU.

        Returns:
            None

        """

        self.lp = language_processor
        self.data = self.lp.data
        self.device = device
        self.append_eos = append_eos
        self.start_token_id = self.lp.start_token_id
        self.end_token_id = self.lp.end_token_id
        self.pad_token_id = self.lp.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        The method to return the pair (input,target) words, after encoding them

        Params:
            idx : The index corresponding to the pair of words to be returned.
            
        Returns:
            None

        """
        input_word, output_word = self.data[idx]
        
        input_sequence = self.lp.encode_word(input_word,self.lp.source_lang,padding=False,append_eos=self.append_eos)
        output_sequence = self.lp.encode_word(output_word,self.lp.target_lang,padding=False,append_eos=self.append_eos)
        
        return torch.tensor(input_sequence).to(self.device), torch.tensor(output_sequence).to(self.device)

def collate_fn(batch,pad_token_id,device):
    
    input_words, target_words = zip(*batch)
    
    padded_inputs = pad_sequence(input_words, batch_first=True, padding_value=pad_token_id)
    
    padded_targets = pad_sequence(target_words, batch_first=True, padding_value=pad_token_id)
    
    input_lengths = torch.LongTensor([len(seq) for seq in input_words]).to(device)
    target_lengths = torch.LongTensor([len(seq) for seq in target_words]).to(device)
    
    return padded_inputs, padded_targets, input_lengths, target_lengths