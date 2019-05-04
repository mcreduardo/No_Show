# Hospital appointment no-show prediction
# Simple NN using TensorFlow
# April 2019
# Eduardo Moura Cirilo Rocha

# learning rates

import numpy as np
import pandas as pd # load csv data
import numpy as np
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

learning_rates = [0.005, 0.01, 0.03, 0.05, 0.08, 0.15]

acc = []
f1 = []
loss = []
for rate in learning_rates:

    NN = nn.NN_Sigmoid(
        hiddenLayers, numFeatures, numLabels, rate,
        cross_entropy_weight = 4,
        optimizer = "Adam"
    )
    # train
    acc_history, f1_history, cost_history, _, _, _, _ = NN.train(
        100, trainX, trainY 
    )
    NN.close_session()

    acc.append(acc_history)
    f1.append(f1_history)
    loss.append(cost_history)






# Plot loss, accuracy, f1
plt.subplot(121)
for idx, rate in enumerate(learning_rates):
    plt.plot(range(len(loss[idx])),loss[idx], label='lr = '+str(rate))
#plt.title('Cost')
plt.ylabel('Weighted Cross Entropy')
plt.xlabel('Epoch')
plt.ylim(top=1.2)


plt.subplot(122)
for idx, rate in enumerate(learning_rates):
    plt.plot(range(len(f1[idx])),f1[idx], label='lr = '+str(rate))
#plt.title('F1-Score')
plt.ylabel('F1-Score')
plt.xlabel('Epoch')
plt.ylim(bottom=0.3)
plt.gca().legend()

plt.show()