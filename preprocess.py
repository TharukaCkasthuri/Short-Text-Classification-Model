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
import logging
import traceback

logger = logging.getLogger()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')


class Preprocess:
    """
    This is text cleaning steps implementation for intent classifications in chat-bot user utterences.
    """
    def __init__(self, stopwords_file , base_dir='./'):
        self.stopwords_file = stopwords_file
        self.base_dir = base_dir
        self.input_data = input_data

    def _load_stopwords(self):
        """
        Loading the list of predifined stopwords from the text file.
        Return:
        -----------
        stopwords : list of stopwords.

        """
        try:
          file = open(self.stopwords_file,'r')
          stopwords = file.read().split('\n')
          file.close()
          logger.info('Loaded stopwords as: ' % stopwords)
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
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
        logger.info('Text after expanding cotradictions  %s' % expanded_text)
        return expanded_text

    def _normalize_text(self,text):
        """ 
        Remove numerical characters, punctuation marks and additiona spaces from text. 
        Parameters:
        -----------
        text : String
        Return:
        -----------
        result_non_space : text without numerica characters, String
        """
        text_lower = text.lower()
        try:
          non_numeric_text = re.sub("[0-9]", "", text_lower)
          text_no_punc = non_numeric_text.translate(str.maketrans('', '', string.punctuation)).strip()
          result_non_space = re.sub(r'\s+',' ',text_no_punc)
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
          traceback.print_exc()
        return result_non_space


    def _remove_stopwords(self, text):
        """ 
        Remove stop words from the text. 
        Parameters:
        -----------
        text : String
        Return:
        -----------
        filtered_text : text without stop words, String
        """
        try:
          stopwords = self._load_stopwords()
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
          traceback.print_exc()
        text_tokens = text.split(" ")
        filtered_words = [word for word in text_tokens if word not in stopwords]
        filtered_text = ' '.join(filtered_words)
        return filtered_text


    def text_clean(self, input_data, text_column = 'text', lable_column = 'category'):
        """ 
        Cleaning the text column.
        """
        try:
          data_path = os.path.join(self.base_dir,input_data)
          data = load_data(data_path, text_column, lable_column)
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
          traceback.print_exc()
        logger.info('Loaded data from path: %s' % data_path)
        data['fortmatted_text'] = data['text'].apply(lambda x: self._handle_contradictions(x))
        logger.info('Completed expanding contradictions')
        data['fortmatted_text'] = data['fortmatted_text'].apply(lambda x: self._normalize_text(x))
        logger.info('Removed all numerica characters and additional spaces')
        data['fortmatted_text'] = data['fortmatted_text'].apply(lambda x: self._remove_stopwords(x))
        logger.info('Removed stopwords from text data')
        save_data(data_path , data)

    def encode_lables(self, input_data , text_column = 'fortmatted_text', lable_column = 'category'):
        """
        Encoding categorical lables. 
        Parameters:
        -----------
        input_data : File name of input data, String
        text_column : String
        lable_column : Sring
        Return:
        -----------
        train_data : Dataframe, dataframe with x and y axis for training.

        """
        try:
          data_path = os.path.join(self.base_dir, input_data)
          data = load_data(data_path, text_column, lable_column)
          logger.info('Loaded data from path: %s' % data_path)
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
          traceback.print_exc()

        possible_labels = data.category.unique()
        label_dict = {}

        try:
          for index, possible_label in enumerate(possible_labels):
              label_dict[possible_label] = index
          data['label'] = data.category.replace(label_dict)
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
          traceback.print_exc()

        output_dir = '/content/drive/MyDrive/text_classification/train'

        if not os.path.exists(output_dir):
          os.makedirs(output_dir)

        #taking category and label into dataframe.
        label_df = pd.DataFrame(label_dict.items() , columns = ['category' , 'label'])
        label_df_path = os.path.join(output_dir,'labels')
        try:
          save_data(label_df_path,label_df)
          logger.info('Saved the label table, Path : %s' % label_df_path)
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
          traceback.print_exc()
          
        #saving training_data
        train_data_path = os.path.join(output_dir,'train')
        train_data = data[[text_column , 'label']]
        try:
          save_data(train_data_path,train_data)
          logger.info('Saved the train data, Path : %s' % train_data_path)
        except Exception as e:
          logger.info('Exiting due to exception: %s' % e)
          traceback.print_exc()

        return train_data


def save_data(path, dataframe):
    """
    Saving a dataframe.
    """
    try:
      dataframe.to_csv(path , index = False)
    except Exception as e:
      logger.info('Exiting due to exception: %s' % e)
      traceback.print_exc()



def load_data(path, text_column, lable_column):
    """
    Loading the raw data from the dirrectory
    Return:
    -----------
    data : The raw data, a pandas dataframe object.
    """
    data = pd.read_csv(path , usecols = [text_column, lable_column])
    return data         


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Preprocessor",
        description="Data preprocessing for intent classifications of chatbot user utterances")
    parser.add_argument('--stopwords_file', nargs='?', default='./stopwords.txt',
                        help='Text file which contains all stop words')
    parser.add_argument('--base_dir', nargs='?', default='./raw_data/train.csv',
                        help='Path to csv file which contains all training data')
    parser.add_argument('--text_column', type=String, default='text',
                        help='Column name of user utterances'),
    parser.add_argument('--lable_column', type=String , default='category',
                        help='Column name of labels'),
    return parser.parse_args()


