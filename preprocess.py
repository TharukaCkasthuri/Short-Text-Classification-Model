#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 1 16:40:00 2022

@author: tharuka
"""

import pandas as pd
import contractions
import re
import string
import os
import traceback
import torch

import logging
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s' , level=logging.INFO) 


class TextCleaner:
    """
    This is text cleaning steps implementation for intent classifications in chat-bot user utterences.
    """

    def __init__(self, stopwords_file = 'stopwords.txt' , base_dir='./'):
        self.base_dir = base_dir
        self.stopwords_file_path = os.path.join(self.base_dir,stopwords_file)

    def _load_stopwords(self):
        """
        Loading the list of predifined stopwords from the text file.

        Return:
        -----------
        stopwords : list of stopwords.

        """
        try:

          file = open(self.stopwords_file_path,'r')
          stopwords = file.read().split('\n')
          file.close()
          logging.info('Loaded stopwords')
        except Exception as e:
          logging.info('Exiting due to exception: %s' % e)
          traceback.print_exc()

        return stopwords
    
    def _handle_contradictions(self,text):
        """ 
        Remove contradictions from text.

        Parameters:
        -----------
        text : String

        Return:
        -----------
        expanded_text : expanded text string, String
        """
        expanded_words = []    
        for word in text.split():
            expanded_words.append(contractions.fix(word))
        #join words to create a sentence
        expanded_text = ' '.join(expanded_words)
        logging.info('Text after expanding cotradictions  %s' % expanded_text)
        return expanded_text

    def _normalize_text(self,text):
        """ 
        Remove numerical characters, punctuation marks and additiona spaces from text. 

        Parameters:
        -----------
        text : String

        Return:
        -----------
        result_non_space : text without numerical characters, String
        """
        text_lower = text.lower()
        try:
          non_numeric_text = re.sub("[0-9]", "", text_lower)
          text_no_punc = non_numeric_text.translate(str.maketrans('', '', string.punctuation)).strip()
          result_non_space = re.sub(r'\s+',' ',text_no_punc)
        except Exception as e:
          logging.info('Exiting due to exception: %s' % e)
          traceback.print_exc()
        return result_non_space

    def _remove_stopwords(self, text , stopwords):
        """ 
        Remove stop words from the text. 

        Parameters:
        -----------
        text : String

        Return:
        -----------
        filtered_text : text without stop words, String
        """
        text_tokens = text.split(" ")
        filtered_words = [word for word in text_tokens if word not in stopwords]
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    def text_clean(self, input_data, text_column = 'text', lable_column = 'category'):
        """ 
        Cleaning the text column.

        Parameters:
        -----------
        input_data : Dataframe, A pandas dataframe which contains text data.
        text_column : String, name of raw text column, which we need to preprocess.
        label_column : String, column name of the lables.

        Return:
        -----------
        cleaned_data : Dataframe, formatted dataframe
        """
        try:
          stopwords = self._load_stopwords()
        except Exception as e:
          logging.info('Exiting due to exception: %s' % e)
          traceback.print_exc()

        input_data['fortmatted_text'] = input_data['text'].apply(lambda x: self._handle_contradictions(x))
        logging.info('Completed expanding contradictions')
        input_data['fortmatted_text'] = input_data['fortmatted_text'].apply(lambda x: self._normalize_text(x))
        logging.info('Removed all numerical characters and additional spaces')
        input_data['fortmatted_text'] = input_data['fortmatted_text'].apply(lambda x: self._remove_stopwords(x ,stopwords))
        logging.info('Removed stopwords from text data')
        #save_data(data_path , data)
        output_data = input_data[['fortmatted_text',lable_column]]

        return output_data


class Encoder:
    """
    This is text cleaning steps implementation for intent classifications in chat-bot user utterences.
    """

    def __init__(self, base_dir='./'):
        self.base_dir = base_dir

    def encode_lables(self, input_data , text_column = 'fortmatted_text', label_column = 'category'):
        """
        Encoding categorical lables. 

        Parameters:
        -----------
        input_data : Dataframe, a pandas dataframe object contains text data (X) and labels (y - categories).
        text_column : String
        lable_column : Sring

        Return:
        -----------
        X : numpy.ndarray, numpy array with x axis for training.
        y : numpy.ndarray,numpy array with y axis for training.
        label_df : Dataframe, dataframe which contains category and labels.

        """

        possible_labels = input_data.category.unique()
        label_dict = {}

        try:
          for index, possible_label in enumerate(possible_labels):
              label_dict[possible_label] = index
          input_data['label'] = input_data.category.replace(label_dict)
        except Exception as e:
          logging.info('Exiting due to exception: %s' % e)
          traceback.print_exc()
        
        output_dir = os.path.join(self.base_dir,'train_data')

        if not os.path.exists(output_dir):
          os.makedirs(output_dir)

        #taking category and label into dataframe.
        label_df = pd.DataFrame(label_dict.items() , columns = ['category' , 'label'])
        label_df_path = os.path.join(output_dir,'labels')
        try:
          label_df.to_csv(label_df_path , index = False)
          logging.info('Saved the label table, Path : %s' % label_df_path)
        except Exception as e:
          logging.info('Exiting due to exception: %s' % e)
          traceback.print_exc()
          
        #saving training_data
        train_data_path = os.path.join(output_dir,'train')
        train_data = input_data[[text_column , 'label']]
        X = input_data[text_column].values
        y = input_data['label'].values

        try:
          train_data.to_csv(train_data_path , index = False)
          logging.info('Saved the train data, Path : %s' % train_data_path)
        except Exception as e:
          logging.info('Exiting due to exception: %s' % e)
          traceback.print_exc()

        return X, y, label_df

class CustomDataset(torch.utils.data.Dataset):
    """
    Stores uttarance data as PyTorch Dataset.
    """
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        """
        An encoding can have keys such as input_ids and attention_mask. Item is a dictionary which has the same keys as the encoding has 
        and the values are the idxth value of the corresponding key (in PyTorch's tensor format).
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)




