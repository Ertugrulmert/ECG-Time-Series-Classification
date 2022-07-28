# Machine Learning for Healthcare Project 1

This repository contains different deep learning models for classifying ECG time series. 
Our models are trained and tested on the well-known [MIT-BIH Arrythmia Database](https://physionet.org/physiobank/database/mitdb/) and 
on the [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/). For a detailed discussion of the 
models and their performances on the given data we refer to the written report. 

***
## Getting started

We use a [Conda](https://docs.conda.io/en/latest/miniconda.html) environment for dependency management. 
Therefore the minimal requirements are Conda. To create an environment, open an Anaconda prompt and run the following:
```
conda env create -f environment_ml.yml
```
If antivirus related permission issues or os access errors occur, running the Anaconda prompt on administrator mode avoids such issues.

## Add your files
The data input files should be added into a new folder `input/`. 
For our model training and evaluating, we used the same data as the [baselinemodel](https://github.com/CVxTz/ECG_Heartbeat_Classification)
and the same train test split. 

***
## Models
All evaluated models can be found in this package. To import the models to your program use
the following import statements:
```
from models import *
```

The models are `scikeras.Wrappers.KerasClassifier` classes. For a detailed instruction on how to use them the reader 
is referred to the [SciKeras documentation](https://www.adriangb.com/scikeras/stable/). 
The following models are implemented (see `/models/`) in the respective self-explanatory files: 

**CNNs** 
- Baseline
- DoubleConvCNN 
- VanillaCNN


**ResNets**
- ResNetDS
- ResNetSmall
- ResNetStandard

**UNET**
- Unet

**RNNs**
- BaseRNN
- BiDirLSTM
- ConvLSTM
- VanillaLSTM
- VanillaRNN

**Ensembles**
- LogRegEnsemble
- MeanEnsemble
- PretrainedEnsemble
- VotingEnsemble

## Testing and Evaluation
The code that we use for evaluating different models is in the respective subfolders. Further the code for the final 
models are also included. The following gives a brief overview over the corresponding files. 
To reproduce our results from the report, follow the exact order of the tasks and their corresponding scripts
as they are described in the following.
### Task 1
For the steps taken to implement and evaluate the vanilla models, refer to the following files. 
or a demonstration of model performance, **VanillaCNN.ipynb** and **Final_RNN_Based_Models.ipynb** notebooks must be run.

| Code | Description |
| -------------- | --------- |
| /CNNs/VanillaCNN.ipynb | Vanilla CNN training and testing, hyperparameter tuning  |
|/RNNs/Final_RNN_Based_Models.ipynb | RNN based models training and testing | 
|/RNNs/RNN_Models_Hyperparam_Search.ipynb | RNN based models hyperparameter tuning| 


### Task 2
For the steps taken to develop the models improving those of Task 1, refer to the following files. 
For a demonstration of model performance, refer again to **Final_RNN_Based_Models.ipynb** as it contains extended RNN based models as well. For a demosntration of ResNet model performance,  **ResNet.ipynb** notebook must be run.

| Code | Description |
| -------------- | --------- |
| /ResNet/ResNet.ipynb | ResNet training and testing, hyperparameter tuning| 
|/RNNs/Final_RNN_Based_Models.ipynb | RNN based models training and testing | 
|/RNNs/RNN_Models_Hyperparam_Search.ipynb | RNN based models hyperparameter tuning| 

### Task 3
| Code | Description |
| -------------- | --------- |

### Task 4
For transfer learning models to run without issues, having the Task 2 models run and saved already is required. Refer to the notebooks in the Task 2 section. To display ResNet transfer learning model training and performance, run **TransferResNet.ipynb**. To display the training and performance of RNN based models, run **Transfer_Learning_RNN_Based_Models.ipynb** notebook.

| Code | Description |
| -------------- | --------- |
| /CNNs/TransferResNet.ipynb | Transfer Learning for ResNet | 
|/RNNs/Transfer_Learning_RNN_Based_Models.ipynb | Transfer Learning for RNN based models | 

***
## Authors
Mert Ertugrul <br>
Johan Lokna <br>
Nora Schneider