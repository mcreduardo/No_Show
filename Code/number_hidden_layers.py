# Hospital appointment no-show prediction
# Simple NN using TensorFlow
# April 2019
# Eduardo Moura Cirilo Rocha

# compare depth

import numpy as np
import pandas as pd # load csv data
from sklearn.model_selection import train_test_split # split into training and test set
import dataset as ds # load dataset
import matplotlib.pyplot as plt # plotting
import scipy.stats


import NN_Sigmoid as nn
from no_show import k_fold_cross_validation
# load dataset
noShow = ds.import_data_df([ds._FILE_PATHS['merged']])
noShow_X, noShow_y = noShow.iloc[:, :-1].values, noShow.iloc[:, -1].values
noShow_y = np.array([[i] for i in noShow_y])

# separate training and test set randomly keeping classes ratio
trainX, testX, trainY, testY = train_test_split(
    noShow_X, noShow_y, test_size=0.2, random_state=42,
    stratify = noShow_y
)


# Hidden layers
numHiddenLayers = [0,1] 
# number of features
numFeatures = trainX.shape[1]
# number of classes
numLabels = trainY.shape[1]

f1 = []
loss = []
for HL in numHiddenLayers:

    hiddenLayers = []
    for i in range(HL): hiddenLayers.append(10)


    l, acc, f = k_fold_cross_validation(10, hiddenLayers, 500)

    f1.append(f)
    loss.append(l)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    print(m)
    print(se)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h, m

f1_mean=[]
f1_error=[]
for x in f1:
    h, m = mean_confidence_interval(x, confidence=0.95)
    f1_mean.append(m)
    f1_error.append(h)

loss_mean=[]
loss_error=[]
for x in loss:
    h, m = mean_confidence_interval(x, confidence=0.95)
    loss_mean.append(m)
    loss_error.append(h)


print(f1_mean)
print(f1_error)
print(loss_mean)
print(loss_error)


# Plot loss, accuracy, f1
plt.subplot(121)
plt.errorbar(numHiddenLayers[:-2],loss_mean[:-2], yerr=loss_error[:-2], fmt='o-')
#plt.title('Cost')
plt.ylabel('Weighted Cross Entropy')
plt.xlabel('Number Hidden Layers')

plt.subplot(122)
plt.errorbar(numHiddenLayers[:-2],f1_mean[:-2], yerr=f1_error[:-2], fmt='o-')
#plt.title('F1-Score')
plt.ylabel('F1-Score')
plt.xlabel('Number Hidden Layers')

plt.show()