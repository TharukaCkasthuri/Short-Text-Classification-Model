# Short Text Classification Using Deep Learning with Pytorch 

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
1. **Download a dataset:**

   The data used to demostrate the model can be found in following link : https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data

    |id| text | target |
| ------------- | ------------- | ------------- |
| 1  | Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all  |1  |
| 2  | SOOOO PUMPED FOR ABLAZE ???? @southridgelife  | 0  |
| 3  | INEC Office in Abia Set Ablaze - http://t.co/3ImaomknnA  | 1 |
| 4  | Building the perfect tracklist to life leave the streets ablaze  | 0  |

   You can use other datasets if you convert them to the right format. 


3. **Train a model**

   ```
   python3 train.py
   ```