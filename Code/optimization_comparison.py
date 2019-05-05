# Hospital appointment no-show prediction
# Simple NN using TensorFlow
# April 2019
# Eduardo Moura Cirilo Rocha

# compate optimizers and learning rates

import numpy as np
import pandas as pd # load csv data
from sklearn.model_selection import train_test_split # split into training and test set
import dataset as ds # load dataset
import matplotlib.pyplot as plt # plotting

import NN_Sigmoid as nn

# load dataset
noShow = ds.import_data_df([ds._FILE_PATHS['merged']])
noShow_X, noShow_y = noShow.iloc[:, :-1].values, noShow.iloc[:, -1].values
noShow_y = np.array([[i] for i in noShow_y])

# separate training and test set randomly keeping classes ratio
trainX, testX, trainY, testY = train_test_split(
    noShow_X, noShow_y, test_size=0.2, random_state=42,
    stratify = noShow_y
)
# separate training in validation and training set
trainX, valX, trainY, valY = train_test_split(
    trainX, trainY, test_size=0.15, random_state=42,
    stratify = trainY
)

# Hidden layers
hiddenLayers = [10, 10, 5] 
# number of features
numFeatures = trainX.shape[1]
# number of classes
numLabels = trainY.shape[1]

# Adam
NN_Adam = nn.NN_Sigmoid(
    hiddenLayers, numFeatures, numLabels, 0.05,
    cross_entropy_weight = 4,
    optimizer = "Adam"
)
# train
acc_history_Adam, f1_history_Adam, cost_history_Adam, _, _, _, _ = NN_Adam.train(
    500, trainX, trainY 
)
NN_Adam.close_session()

# SGD
NN_SGD = nn.NN_Sigmoid(
    hiddenLayers, numFeatures, numLabels, 0.3,
    cross_entropy_weight = 4,
    optimizer = "GD"
)
# train
acc_history_SGD, f1_history_SGD, cost_history_SGD, _, _, _, _ = NN_SGD.train(
    500, trainX, trainY 
)
NN_SGD.close_session()



# Plot loss, accuracy, f1
plt.subplot(121)
plt.plot(range(len(cost_history_Adam)),cost_history_Adam, label='Adam, lr = 0.05')
plt.plot(range(len(cost_history_SGD)),cost_history_SGD, label='SGD, lr = 0.3')
#plt.title('Cost')
plt.ylabel('Weighted Cross Entropy')
plt.xlabel('Epoch')


plt.subplot(122)
plt.plot(range(len(f1_history_Adam)),f1_history_Adam, label='Adam, lr = 0.05')
plt.plot(range(len(f1_history_SGD)),f1_history_SGD, label='SGD, lr = 0.3')
#plt.title('F1-Score')
plt.ylabel('F1-Score')
plt.xlabel('Epoch')
plt.gca().legend(('Adam, lr = 0.05','SGD, lr = 0.3'))
plt.show()