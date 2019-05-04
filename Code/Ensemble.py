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
from sklearn.metrics import accuracy_score


from no_show import Suppressor # supress std output

#with Suppressor():

numModels = 20

# load dataset
noShow = ds.import_data_df([ds._FILE_PATHS['merged']])
noShow_X, noShow_y = noShow.iloc[:, :-1].values, noShow.iloc[:, -1].values
noShow_y = np.array([[i] for i in noShow_y])

# separate training and test set randomly keeping classes ratio
trainX, testX, trainY, testY = train_test_split(
    noShow_X, noShow_y, test_size=0.2, random_state=42,
    stratify = noShow_y
)

# number of features
numFeatures = trainX.shape[1]
# number of classes
numLabels = trainY.shape[1]

NN = []
for i in range(numModels):

    # separate training in validation and training set
    trainX, valX, trainY, valY = train_test_split(
        trainX, trainY, test_size=0.15, random_state=42,
        stratify = trainY
    )

    # Hidden layers
    hiddenLayers = [10, 5] 

    net = nn.NN_Sigmoid(
        hiddenLayers, numFeatures, numLabels, 0.05,
        cross_entropy_weight = 4,
        optimizer = "Adam"
    )

    # train
    net.train(
        500, trainX, trainY, 
        valX=valX, valY=valY, val_epochs=25, val_patience=10
    )

    NN.append(net)


predictions = np.around(testY/1000)
labels = np.around(testY/1000)
for net in NN:
    p,_,_,_ = net.predict(
        testX, testY
    )
    l = np.around(p)
    predictions += p
    labels += l

predictions = predictions/numModels
labels = labels/numModels
print(predictions)
print(labels)


print(accuracy_score(np.around(predictions), testY))
print(accuracy_score(np.around(labels), testY))



