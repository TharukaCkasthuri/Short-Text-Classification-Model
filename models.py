#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:52:00 2022

@author: tharuka
"""
import torch
#import torch.nn as nn
from transformers import BertModel

import logging
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s' , level=logging.INFO) 

class BertWithLSTMClassifier(torch.nn.Module):
    """
    A pre-trained BERT model with a LSTM layer for classification.
    """
    
    def __init__(self , n_classes, model_name):
        super(BertWithLSTMClassifier, self).__init__()
        self.n_classes = n_classes
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.lstm = torch.nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=False)
        self.linear = torch.nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, tokens, attention_mask):
        """
        """
        output = self.bert(tokens, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        sequence_output = output.last_hidden_state
        lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
        hidden = torch.cat((lstm_output[:,-1, :self.hidden_size],lstm_output[:,0, self.hidden_size:]),dim=-1)
        linear_output = self.linear(hidden.view(-1,self.hidden_size)) 
        ### only using the output of the last LSTM cell to perform classification
        return linear_output
        
    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithLSTMClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=False
    
    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithLSTMClassifier is trained
        both the wieghts of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=True

class BertWithBiLSTMClassifier(torch.nn.Module):
    """
    A pre-trained BERT model with a Bidirectional LSTM layer for classification.
    """    
    def __init__(self , n_classes, model_name):
        super(BertWithBiLSTMClassifier, self).__init__()
        self.n_classes = n_classes
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.lstm = torch.nn.LSTM(768, self.hidden_size, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(self.hidden_size*2, self.n_classes)

    def forward(self, tokens, attention_mask):
        output = self.bert(tokens, attention_mask=attention_mask)
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        sequence_output = output.last_hidden_state
        lstm_output, (h,c) = self.lstm(sequence_output) ## extract the 1st token's embeddings
        hidden = torch.cat((lstm_output[:,-1, :self.hidden_size],lstm_output[:,0, self.hidden_size:]),dim=-1)
        linear_output = self.linear(hidden.view(-1,self.hidden_size*2)) 
        return linear_output


        
    def freeze_bert(self):
        """
        Freezes the parameters of BERT so when BertWithLSTMClassifier is trained
        only the wieghts of the custom classifier are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=False
    
    def unfreeze_bert(self):
        """
        Unfreezes the parameters of BERT so when BertWithLSTMClassifier is trained
        both the wieghts of the custom classifier and of the underlying BERT are modified.
        """
        for param in self.bert.named_parameters():
            param[1].requires_grad=True


            
