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


class TextCleaner:
    """
    This is text cleaning steps implementation for intent classifications in chat-bot user utterences.
    """
    def __init__(self, stopwords_file , data_path, save_path, text_column = 'text' , lable_column = 'category'):
        self.stopwords_file = stopwords_file
        self.data_path = data_path
        self.save_path = save_path
        self.text_column = text_column
        self.lable_column = lable_column

    def _load_stopwords(self):
        """
        Loading the list of predifined stopwords from the text file.
        Return:
        -----------
        stopwords : list of stopwords.

        """
        file = open(self.stopwords_file,'r')
        stopwords = file.read().split('\n')
        file.close()
        return stopwords
    
    def _handle_contradictions(text):
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
        return expanded_text

    def _normalize_text(text):
        """ 
        Remove numerical characters, punctuation marks and additiona spaces from text. 
        Parameters:
        -----------
        text : String
        Return:
        -----------
        expanded_text : text without numerica characters, String
        """
        text_lower = text.lower()
        non_numeric_text = re.sub("[0-9]", "", text_lower)
        text_no_punc = non_numeric_text.translate(str.maketrans('', '', string.punctuation)).strip()
        result_non_space = re.sub(r'\s+',' ',text_no_punc)
        return result_non_space


    def _remove_stopwords(self, text):
        """ 
        Remove stop words from the text. 
        Parameters:
        -----------
        text : String
        Return:
        -----------
        expanded_text : text without stop words, String
        """
        stopwords = self._load_stopwords()
        text_tokens = text.split(" ")
        filtered_words = [word for word in text_tokens if word not in stopwords]
        filtered_text = ' '.join(filtered_words)
        return filtered_text

    def text_clean(self):
        """ 
        Cleaning the text column.
        """
        data = load_data(self.data_path, self.text_column, self.lable_column)
        data['fortmatted_text'] = data['text'].apply(lambda x: self._handle_contradictions(x))
        data['fortmatted_text'] = data['fortmatted_text'].apply(lambda x: self._normalize_text(x))
        data['fortmatted_text'] = data['fortmatted_text'].apply(lambda x: self._remove_stopwords(x))

        save_data(self.save_path , data)


class LabelMaker:
    """
    """

    def __init__(self, data_path, save_path, text_column = 'fortmatted_text', lable_column = 'category'):
        self.data_path = data_path
        self.text_column = text_column
        self.lable_column = lable_column


def save_data(path, dataframe):
    """
    Saving a dataframe.
    """
    dataframe.to_csv(path , index = False)


def load_data(path, text_column, lable_column):
    """
    Loading the raw data from the dirrectory
    Return:
    -----------
    raw_data : The raw data, a pandas dataframe object.
    """
    raw_data = pd.read_csv(path , usecols = [text_column, lable_column])
    return raw_data


    

def parse_args():
    parser = argparse.ArgumentParser(
        prog="Preprocessor",
        description="Data preprocessing for intent classifications of chatbot user utterances")
    parser.add_argument('--stopwords_file', nargs='?', default='./stopwords.txt',
                        help='Text file which contains all stop words')
    parser.add_argument('--data_path', nargs='?', default='./raw_data/train.csv',
                        help='Path to csv file which contains all training data')
    parser.add_argument('--text_column', type=String, default='text',
                        help='Column name of user utterances'),
    parser.add_argument('--lable_column', type=String , default='category',
                        help='Column name of labels'),
    parser.add_argument('--save_path', nargs="?" , default = './train_data')
    return parser.parse_args()


