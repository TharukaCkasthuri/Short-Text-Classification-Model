#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:14:00 2022

@author: tharuka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import argparse
import os
import traceback
import torch

import preprocess
import models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW

from models import BertWithLSTMClassifier

import logging
logging.basicConfig(format='%(name)s - %(asctime)s - %(levelname)s - %(message)s' , level=logging.INFO) 


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Intent Classifier",
        description="This is to train intent classification model")
    parser.add_argument('--classifier', nargs='?', default='BertWithBiLSTMClassifier'
                        help='name of the classifier model.')
    parser.add_argument('--stopwords_file', nargs='?', default='stopwords.txt',
                        help='file name of the stopwords.')
    parser.add_argument('--base_dir', nargs="?", default='./',
                        help='path to the base dirrectory.'),
    parser.add_argument('--input_data', nargs="?", default='raw_data/train.csv',
                        help='name of the input data file.'),
    parser.add_argument('--label_column', nargs="?" , default='category',
                        help='name of the label column.'),
    parser.add_argument('--text_column', nargs="?", default='text',
                        help='name of the text column.'),
    parser.add_argument('--bert_model', nargs="?",  default='bert-base-cased',
                        help='bert vertion to be used.' ),
    parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')

    return parser.parse_args()

def prepare_data(base_dir, input_data, text_column, label_column, stopwords_file, bert_model, hparams):
    """
    Preprocessing data to feed into nueral network.

    Parameters:
    -----------
    base_dir: 
    input_data: 
    text_column:
    label_column:
    stopwords_file:
    bert_model:
    hparams:
    Return:
    -----------
    train_loader: 
    val_loader:
    """
    try:
        input_data_path = os.path.join(base_dir,input_data)
        data = pd.read_csv(input_data_path , usecols= [text_column, label_column])
    except ValueError as ve:
        print('Exiting due to exception: %s' % ve)
        print('Make sure the entered column names are correct!')
        traceback.print_exc()
    except FileNotFoundError as fe:
        print('Exiting due to exception: %s' % fe)
        print('Make sure the entered path to the data set is correct!')
        traceback.print_exc()
    except Exception as e:
        logging.info('Exiting due to exception: %s' % e)
        print('Exiting due to exception: %s' % e)
        traceback.print_exc()

    cleaner = preprocess.TextCleaner(stopwords_file)
    cleaned_data = cleaner.text_clean(data)
    encoder = preprocess.Encoder(base_dir)
    X,y, label_df = encoder.encode_lables(cleaned_data , label_column = label_column)
    train_x, val_x, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info('Size of train data (# of entries): %s', len(train_x))
    logging.info('Size of validation data (# of entries): %s ', len(val_x))
    logging.info('Downloading  %s ', bert_model)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    train_encodings = tokenizer(train_x.tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_x.tolist(), truncation=True, padding=True)
    logging.info('Training and validation data has been encoded with %s ', bert_model)
    train_dataset = preprocess.CustomDataset(train_encodings, train_y)
    val_dataset = preprocess.CustomDataset(val_encodings, val_y)
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=True)
    logging.info('Created train & val datasets.')  

    return train_loader,val_loader

def eval_prediction(y_batch_actual, y_batch_predicted):
    """Return batches of accuracy and f1 scores.

    Parameters:
    -----------
    y_batch_actual: actual labels of the batch.
    y_batch_predicted: predicted lables of the batch.

    Return:
    -----------
    acc: float, accuracy score.
    f1: float, f1 score.
    """
    y_batch_actual_np = y_batch_actual.cpu().detach().numpy()
    y_batch_predicted_np = np.round(y_batch_predicted.cpu().detach().numpy())
    acc = accuracy_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np)
    f1 = f1_score(y_true=y_batch_actual_np, y_pred=y_batch_predicted_np, average='weighted')
    
    return acc, f1

def training_step(dataloader, model, optimizer, loss_fn, if_freeze_bert):
    """Method to train the model.

    Parameters:
    -----------
    dataloader: training batch data.
    model: classification model.
    optimizer: optimizer.
    loss_fn: loss function.
    if_freeze_bert: Boolean value.
    """
    
    model.train()
    model.freeze_bert() if if_freeze_bert else model.unfreeze_bert()
    epoch_loss = 0
    size = len(dataloader.dataset)
 
    for i, batch in enumerate(dataloader):        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
    
        outputs = model(tokens=input_ids, attention_mask=attention_mask)
                        
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels.long())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

def evaluation_step(dataloader, model, loss_fn):
    """Method to test the model's accuracy and loss on the validation set.

    Parameters:
    -----------
    dataloader:
    model:
    loss_fn:

    Return:
    -----------
    acc:
    f1:
    loss:
    """
    model.eval()
    model.freeze_bert()
    size = len(dataloader)
    f1, acc, loss = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = batch['labels'].to(device)  
            pred = model(tokens=X, attention_mask=attention_mask)
            big_val, big_idx = torch.max(pred.data, dim=1)
            acc_batch, f1_batch = eval_prediction(y.float(), big_idx) 
            loss_batch = loss_fn(pred, y.long())                       
            acc += acc_batch
            f1 += f1_batch
            loss += loss_batch

        acc = round(acc/size, 2)
        f1 = round(f1/size, 2)
        loss = round(loss.item()/size, 2)
                
    return acc, f1, loss

def train(base_dir, train_loader, val_loader, model, num_of_epochs, optimizer, loss_fn, if_freeze_bert):
    """
    """

    tqdm.pandas()
    best_acc, best_f1 = 0, 0
    model_file_name = str(time.time()) + "_trained_model.pt"
    path = os.path.join(base_dir,model_file_name)

    train_f1s = []
    validation_f1s = []
    train_losses = []
    validation_losses = []

    for i in tqdm(range(num_of_epochs)):
        print("Epoch: #{}".format(i+1))
        if i < 10:
            if_freeze_bert = False
            print("Bert is not freezed")
        else:
            if_freeze_bert = True
            print("Bert is freezed")
        
        training_step(train_loader, model,optimizer, loss_fn, if_freeze_bert)

        train_acc, train_f1, train_loss = evaluation_step(train_loader, model, loss_fn)
        val_acc, val_f1, val_loss = evaluation_step(val_loader, model, loss_fn)
        train_f1s.append(train_f1.item())
        validation_f1s.append(val_f1.item())
        train_losses.append(train_loss.item())
        validation_losses.append(validation_loss.item())
        
        #print("Training results: ")
        #print("Acc: {:.3f}, f1: {:.3f}".format(train_acc, train_f1))
        logging.info("Training Loss : {loss} Training Accuracy : {acc} Training F1 Score: {f1}".format(loss=train_loss, acc=train_acc, f1=train_f1))

        #print("Validation results: ")
        #print("Acc: {:.3f}, f1: {:.3f}".format(val_acc, val_f1))
        logging.info("Validation Loss : {loss} Validation Accuracy : {acc} Validation F1 Score: {f1}".format(loss=val_loss, acc=val_acc, f1=val_f1))

        
        if val_acc > best_acc:
            best_acc = val_acc    
            torch.save(model, path)

    plt.figure(figsize=(12,6)) 
    plt.plot(range(1, num_of_epochs+1), train_f1s, 'g', label='Training F1 Score Curve')
    plt.plot(range(1, num_of_epochs+1), validation_f1s, 'r', label='Validation F1 Score Curve')
    plt.plot(range(1, num_of_epochs+1), train_losses, 'b', label='Training Loss Curve')
    plt.plot(range(1, num_of_epochs+1), validation_losses, 'c', label='Validation Loss Curve')
    plt.title("Training/Validation F1 Score/Loss Curves")
    plt.legend()
    plot_file = str(time.time())+"_plot.png"
    #plot_path = os.path.join(base_dir,plot_file)
    plt.savefig(os.path.join(base_dir,plot_file))

def setup():
    """
    Check the available GPU and use it if it is exist. Otherwise use CPU.
    """
    if torch.cuda.is_available():        
        device = torch.device("cuda")
        logging.info('Running on GPU: %s', torch.cuda.get_device_name(0))
        print('Running on GPU: s', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logging.info('Running on CPU.')
        print('Running on CPU.')
    return device

if __name__ == '__main__':
    start_time = time.time() 
    args = parse_args()
    classifier = args.classifier
    stopwords_file = args.stopwords_file
    input_data = args.input_data
    base_dir = args.base_dir
    label_column = args.label_column
    text_column = args.text_column
    bert_model = args.bert_model

    hparams = {"batch_size": 32,
                "num_of_epochs": 15,
                "learning_rate":2e-5}

    train_data, validation_data = prepare_data(base_dir=base_dir , 
                    input_data= input_data, 
                    text_column= text_column, 
                    label_column=label_column, 
                    stopwords_file=stopwords_file , 
                    bert_model = bert_model,
                    hparams = hparams)

    device = setup()

    if classifier == 'BertWithLSTMClassifier':
        cls_model = models.BertWithLSTMClassifier(77,bert_model)
    else if classifier == 'BertWithBiLSTMClassifier'
        cls_model = models.BertWithBiLSTMClassifier(77,bert_model)
    else:
        pass
    cls_model.to(device)

    optimizer = AdamW(cls_model.parameters(), lr=hparams["learning_rate"])
    logging.info('Initialized optimizer.')
    print('Initialized optimizer.')
    loss_fn = torch.nn.CrossEntropyLoss()
    logging.info('Initialized optimizer.')
    print('Initialized loss function.')

    train(base_dir, train_data, validation_data, cls_model, 20, optimizer, loss_fn, False)











