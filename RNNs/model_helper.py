
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import backend as K
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

import seaborn as sns
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix



"""
Standard Training and Evaluation Function
This function is used to demonstrate the performance of the final verisons
of models.
"""

def train_test_model( model, X_train, Y_train, X_test, Y_test,
                      epochs=100, batch=200, binary_task=True, 
                      train=True, save_name = ""):

    """Standard Training and Evaluation Function
    This function is used to demonstrate the performance of the final verisons
    of models.

    Keyword arguments:
    model -- the keras model to be trained and tested
    X_train, Y_train -- training dataset
    X_test, Y_test -- testing dataset
    epochs -- number of epochs for training (default 100)
    batch -- mini-batch size for training (default 200)
    binary_task -- whether the model is a binary
                   or multiclass classifier (default True)
    train -- whether to train first or try to load model first (default True)
    save_name -- file name for saving the final model (default "")

    """

    #callbacks to stop or change learning rate when held out validation set loss 
    #stops improving, patience selected high due to instability of RNNs
    early = EarlyStopping(monitor="val_loss", patience=15, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", patience=7, verbose=1)
    if save_name:
        checkpoint = ModelCheckpoint(filepath=save_name, monitor='val_loss', verbose=1, save_best_only=True) 
        callbacks_list = [checkpoint, early, redonplat] 
    else:
        callbacks_list = [early, redonplat] 
    #callbacks_list = [early, redonplat] 

    if not train and save_name:
    	model = keras.models.load_model(save_name)
    else:

        if isinstance(model, tf.keras.Sequential):

            history_callback = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, 
                                 verbose=2, validation_split=0.1, callbacks=callbacks_list)
        else:
            history_callback = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, 
                                 verbose=2, validation_split=0.1)


    # format of results and metrics chosen according to dataset type/class num.
    if binary_task:
        pred_test = model.predict(X_test)
        pred_test = (pred_test>0.5).astype(np.int8)
    
        print("Test f1 score : %s "% f1_score(Y_test, pred_test) )
 
        print("Test accuracy score : %s "% accuracy_score(Y_test, pred_test))
        
        print("Test AUROC score : %s "% roc_auc_score(Y_test, pred_test))
        
        print("Test AUPRC score : %s "% average_precision_score(Y_test, pred_test))
    else:
        pred_test = model.predict(X_test)

        if len(pred_test.shape) > 1:
            pred_test = np.argmax(pred_test, axis=-1)

        print("Test f1 score : %s "% f1_score(Y_test, pred_test, average="macro"))

        print("Test accuracy score : %s "% accuracy_score(Y_test, pred_test))

    #if save_name and train:
        #model.model().save(save_name + ".h5")
        
        #with open(save_name +"_history"+ ".pkl" , 'wb') as f:
        #    pickle.dump(history_callback.history, f)
            
    return model


def train_test_multiple( model, X_train, Y_train, X_test, Y_test,
                      epochs=100, batch=200, binary_task=True, k=5):


    """Multiple Trial Training and Evaluation Function 
    This function is used to train and test a model k times and
    obtain average evaulation metrics and standard deviations.

    Keyword arguments:
    model -- the keras model to be trained and tested
    X_train, Y_train -- training dataset
    X_test, Y_test -- testing dataset
    epochs -- number of epochs for training (default 100)
    batch -- mini-batch size for training (default 200)
    binary_task -- whether the model is a binary
                   or multiclass classifier (default True)
    k -- number of times to traind and test model

    """

    #arrays to store evaluation metric results for each run
    accuracies = np.zeros(k)
    f1s = np.zeros(k)
    if binary_task:
        AUROCs = np.zeros(k)
        AUPRCs = np.zeros(k)

    #training and testing k times
    for i in range(k):


        print("Model Train&Test Trial - ",i+1)

        
        #callbacks to stop or change learning rate when held out validation set loss 
        #stops improving, patience selected high due to instability of RNNs
        early = EarlyStopping(monitor="val_loss", patience=15, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_loss", patience=7, verbose=1)
        callbacks_list = [early, redonplat] 

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch, 
                                     verbose=1, validation_split=0.1, callbacks=callbacks_list )

        # format of results and metrics chosen according to dataset type/class num.
        if binary_task:
            pred_test = model.predict(X_test)
            pred_test = (pred_test>0.5).astype(np.int8)
        
            print("Test f1 score : %s "% f1_score(Y_test, pred_test) )
            f1s[i] = f1_score(Y_test, pred_test)

            print("Test accuracy score : %s "% accuracy_score(Y_test, pred_test))
            accuracies[i] = accuracy_score(Y_test, pred_test)

            print("Test AUROC score : %s "% roc_auc_score(Y_test, pred_test))
            AUROCs[i] = roc_auc_score(Y_test, pred_test)

            print("Test AUPRC score : %s "% average_precision_score(Y_test, pred_test))
            AUPRCs[i] = average_precision_score(Y_test, pred_test)

        else:
            pred_test = model.predict(X_test)

            if len(pred_test.shape) > 1:
                pred_test = np.argmax(pred_test, axis=-1)

            print("Test f1 score : %s "% f1_score(Y_test, pred_test, average="macro"))
            f1s[i] = f1_score(Y_test, pred_test, average="macro")
            
            print("Test accuracy score : %s "% accuracy_score(Y_test, pred_test))
            accuracies[i] = accuracy_score(Y_test, pred_test)
    
    print("All trials completed.")
    if binary_task:
         
        print("Average Test f1 score : ", np.mean(f1s) ,"std : ", np.std(f1s) )

        print("Average Test accuracy score : ", np.mean(accuracies) ,"std : ", np.std(accuracies) )

        print("Average Test AUROC score : ", np.mean(AUROCs) ,"std : ", np.std(AUROCs) )

        print("Average Test AUPRC score : ", np.mean(AUPRCs) ,"std : ", np.std(AUPRCs) )

        return (f1s,accuracies,AUROCs,AUPRCs)

    else:

        print("Average Test f1 score : ", np.mean(f1s) ,"std : ", np.std(f1s) )

        print("Average Test accuracy score : ", np.mean(accuracies) ,"std : ", np.std(accuracies) )
        
        return (f1s,accuracies)