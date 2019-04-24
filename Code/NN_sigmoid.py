# Hospital appointment no-show prediction
# Simple NN using TensorFlow
# April 2019
# Eduardo Moura Cirilo Rocha

import tensorflow as tf
import pandas as pd # load csv data
import numpy as np
import matplotlib.pyplot as plt # plotting

from sklearn.model_selection import train_test_split # split into training and test set
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score

import dataset as ds


# Load data #######################################################################################

noShow = ds.import_data_df([ds._FILE_PATHS['merged']])
noShow_X, noShow_y = noShow.iloc[:, :-1].values, noShow.iloc[:, -1].values
noShow_y = np.array([[i] for i in noShow_y])
#noShow_y= pd.get_dummies(noShow_y).values


trainX, testX, trainY, testY = train_test_split(
    noShow_X, noShow_y, test_size=0.33, random_state=42,
    stratify = noShow_y
)




# Training parameters #############################################################################

# Number of Epochs in our training
numEpochs = 10000

# Hidden layers
hiddenLayers = [50, 50, 30, 10] 
numHiddenLayers = len(hiddenLayers)

# Defining our learning rate iterations (decay)
'''
learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)
'''
learning_rate = 0.01 # AdamOptimizer below

# Output
printOutput = True
plotLoss = True
printConfusionMatrix = True
test_set_result_path = 'Results/test_set_result.csv'
plotPRCurve = True

###################################################################################################

# number of features
numFeatures = trainX.shape[1]

# number of classes
numLabels = trainY.shape[1]

# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

is_training=tf.Variable(True,dtype=tf.bool)

# weights init
initializer = tf.contrib.layers.xavier_initializer()


###################################################################################################
# Model

layers = [] # variable size for number of hidden layers

# input layer
input_layer = X
layers.append(input_layer)

# hidden layers (ReLU activation)
for i in range(numHiddenLayers):
    h = tf.layers.dense(layers[-1], hiddenLayers[i], activation=tf.nn.relu,
        kernel_initializer=initializer)
    layers.append(h)


# output layer
output_layer = tf.layers.dense(layers[-1], numLabels, activation=None)
predicted = tf.nn.sigmoid(output_layer)
correct_pred = tf.equal(tf.round(predicted), yGold)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# loss
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=yGold, logits=output_layer)
weighted_cross_entropy = tf.nn.weighted_cross_entropy_with_logits(pos_weight = 4,targets=yGold, logits=output_layer)
cost = tf.reduce_mean(weighted_cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


###################################################################################################
# session

# track cost
acc_history = np.empty(shape=[1],dtype=float)
cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for step in range(numEpochs):
        sess.run(optimizer, feed_dict={X: trainX, yGold: trainY})
        loss, _, acc = sess.run([cost, optimizer, accuracy],
            feed_dict={X: trainX, yGold: trainY})
        acc_history = np.append(acc_history, acc)
        cost_history = np.append(cost_history, loss)
        if step % int(numEpochs/20) == 0 and printOutput:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
            
    # Test model and check accuracy
    if printOutput:
        print('Test Accuracy:', sess.run(accuracy,
            feed_dict={X: testX, yGold: testY}))
    
    # Save test results to file
    test_predicted = sess.run(tf.cast(tf.round(predicted), tf.int32),
        feed_dict={X: testX})
    test_predicted_score = sess.run(predicted,
        feed_dict={X: testX})
    #print(testY)
    test_actual = testY
    #results = pd.DataFrame({'Predicted':test_predicted,'Actual':test_actual})
    #results.to_csv(test_set_result_path, index=False)

###################################################################################################

# Confusion Matrix
if printConfusionMatrix:
    print(confusion_matrix(test_actual, test_predicted))

# Plot loss and accuracy
if plotLoss:
    plt.subplot(121)
    plt.plot(range(len(cost_history)),cost_history)
    plt.title('Cost')

    plt.subplot(122)
    #plt.plot(range(len(acc_history)),acc_history)
    plt.title('Accuracy')
    plt.axis([0,numEpochs,max(acc_history)-0.05,max(acc_history)])

    plt.show()

# Plot precision/recall
if plotPRCurve:
    precision, recall, _ = precision_recall_curve(test_actual, test_predicted_score)
    average_precision = average_precision_score(test_actual, test_predicted_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    plt.show()