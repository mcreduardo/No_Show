# Hospital appointment no-show prediction
# Simple NN using TensorFlow
# April 2019
# Eduardo Moura Cirilo Rocha

import pandas as pd # load csv data
import numpy as np
from sklearn.model_selection import train_test_split # split into training and test set
from sklearn.model_selection import StratifiedKFold
import dataset as ds # load dataset
import NN_Sigmoid as nn # implementation of DecisionTree

# suppress std out
import sys, traceback, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
class Suppressor(object):
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
    def write(self, x): pass

# test different architectures 

# Good learning rate?

# k-fold cross validation
def k_fold_cross_validation(k, hiddenLayers, numEpochs):

    # load dataset
    noShow = ds.import_data_df([ds._FILE_PATHS['merged']])
    noShow_X, noShow_y = noShow.iloc[:, :-1].values, noShow.iloc[:, -1].values
    noShow_y = np.array([[i] for i in noShow_y])

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    skf.get_n_splits(noShow_X, noShow_y)
    #print(skf)  

    # store results
    losses = []
    accuracies = []
    f1s = []

    fold = 1
    for train_index, test_index in skf.split(noShow_X, noShow_y):
        #print("Fold: "+str(fold))
        fold += 1
        trainX, testX = noShow_X[train_index], noShow_X[test_index]
        trainY, testY = noShow_y[train_index], noShow_y[test_index]

        # separate training in validation and training set
        trainX, valX, trainY, valY = train_test_split(
            trainX, trainY, test_size=0.15, random_state=42,
            stratify = trainY
        )

        # number of features
        numFeatures = trainX.shape[1]
        # number of classes
        numLabels = trainY.shape[1]

        # init
        NN = nn.NN_Sigmoid(
            hiddenLayers, numFeatures, numLabels, 
            learning_rate = 0.05,
            cross_entropy_weight = 4,
            optimizer = "Adam"
        )
        # train
        NN.train(
            numEpochs, trainX, trainY, 
            valX=valX, valY=valY, val_epochs=25, val_patience=5
        )
        # test
        _, loss, acc, f1 = NN.predict(
            testX, testY
        )
        # close tf session
        NN.close_session()

        losses.append(loss)
        accuracies.append(acc)
        f1s.append(f1)
        
    return losses, accuracies, f1s




if __name__=="__main__":

    with Suppressor():
        hiddenLayers = [20, 20,20,20,10,5]
        numEpochs = 500
        k = 10
        loss, acc, f1 = k_fold_cross_validation(k, hiddenLayers, numEpochs)
    
    print("\nResults of cross validation:")
    mean = np.mean(np.array(loss))
    stddev = np.std(np.array(loss))
    print("Loss:\tMean: %.4f\tStddev: %.8f" % (mean, stddev))
    mean = np.mean(np.array(acc))
    stddev = np.std(np.array(acc))
    print("Acc:\tMean: %.4f\tStddev: %.8f" % (mean, stddev))
    mean = np.mean(np.array(f1))
    stddev = np.std(np.array(f1))
    print("F1:\tMean: %.4f\tStddev: %.8f" % (mean, stddev))

