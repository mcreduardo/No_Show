# Hospital appointment no-show prediction
# Simple NN using TensorFlow
# April 2019
# Eduardo Moura Cirilo Rocha

import pandas as pd # load csv data
import numpy as np
from sklearn.model_selection import train_test_split # split into training and test set

import dataset as ds # load dataset
import NN_Sigmoid as nns # implementation of DecisionTree



noShow = ds.import_data_df([ds._FILE_PATHS['merged']])
noShow_X, noShow_y = noShow.iloc[:, :-1].values, noShow.iloc[:, -1].values
noShow_y = np.array([[i] for i in noShow_y])
#noShow_y= pd.get_dummies(noShow_y).values

# separate training and test set randomly keeping classes ratio
trainX, testX, trainY, testY = train_test_split(
    noShow_X, noShow_y, test_size=0.2, random_state=42,
    stratify = noShow_y
)

# separate training in validation and training set
trainX, valX, trainY, valY = train_test_split(
    trainX, trainY, test_size=0.2, random_state=42,
    stratify = noShow_y
)




# Hidden layers
hiddenLayers = [10] 
# number of features
numFeatures = trainX.shape[1]
# number of classes
numLabels = trainY.shape[1]


# init
NN = nns.NN_Sigmoid(hiddenLayers, numFeatures, numLabels, 0.01,
    cross_entropy_weight = 4,
    optimizer = "Adam")

# train
NN.train(
        150, 0.01, trainX, trainY, 
        valX=testX, valY=testY, val_epochs=50
    )


# test
NN.predict(
    testX, testY
)

# save session
NN.save_session("Saved_sessions/model.ckpt")


# close tf session
NN.close_session()

'''
# create new untrained NN
NN2 = nns.NN_Sigmoid(hiddenLayers, numFeatures, numLabels, 0.01,
    cross_entropy_weight = 4,
    optimizer = "Adam")

# load session
NN2.load_session("Saved_sessions/model.ckpt")

# test
NN2.predict(
    testX, testY
)
'''