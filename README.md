# Short Text Classification Using BERT with Pytorch 

## This is a project developed for short text intent classification in banking conversational agents. 

The main objective of this repository is to demostrate the effect of various embedding techniques and classification algorithms for short text classification by implementing several models with coded in PyTorch. In order to provide a better understanding of the models, I have used banking data from the following <a href="https://github.com/PolyAI-LDN/task-specific-datasets">Github Repository</a>

### Installing dependencies

1. Install Python 3.

2. Install the latest pytorch version with GPU support.

3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

### Training 

1. **Download the dataset:**

   The data used to demostrate the model can be found in following link : https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data , and extract the data into the 'raw_data' folder.

   You can use other datasets if you convert them to the right format. 

2. **Preprocessing Steps**
   I'm using hugging face library as BERT implementation so an additional consideration to take into account is that Hugging Face's tokenizer employs subword tokenization as detailed in their summary here.
    *   Fixing the negation of some of the auxiliary verbs (eg.: shouldnt -> should not) and some of the personal pronouns (eg.: im -> i am)
    *   Replacing the special characters with apropriate words (Ex: & - and )
    *   Removing numerical charaters.
    *   Removing additional spaces.

3. **Train a model**

Models list : 
    *   Bert with LSTM Layer for classification 
    *   Bert with BiLSTM Layer for classification 
    *   Bert with CNN Layer for classification 

   ```
   python3 train.py
   ```

Can change some parameters like this,
```
 train.py [-h] [--classifier NAME OF CLASSIFICATION MODEL] [--stopwords_file STOPWORD_FILE]
         [--base_dir BASE_DIR] [--input_data TRAINING_DATA]
         [--label_column LABEL_COLUMN_NAME] [--text_column TEXT_COLUMN_NAME]
         [--bert_model BERT_VERSION] [--hparams HYPERPARAMETERS_DICT]
```

4.  **Evaluating a model**
5.  **Making Inference**




