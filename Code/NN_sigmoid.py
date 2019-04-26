# Simple NN using TensorFlow
# April 2019
# Eduardo Moura Cirilo Rocha

import tensorflow as tf
#import pandas as pd # load csv data
import numpy as np
#import matplotlib.pyplot as plt # plotting

#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_curve
#from sklearn.utils.fixes import signature
#from sklearn.metrics import average_precision_score

import dataset as ds


class NN_Sigmoid:
    """
    Simple NN for binary classification using Sigmoid function.
    Fully connected layers with ReLU activation.
    """

    ####################################
    def __init__(
        self, hiddenLayers, numFeatures, numLabels, learning_rate, 
        cross_entropy_weight = 1,
        optimizer = "GD"
    ):
        """
        Constructor: Build network

        "hiddenLayers" contains the structure of the network to be trained:
            list with number of neurons per hidden layer.

        "numFeatures" is the number of features

        "numLabels" is the number of labels

        "learning_rate" is the learning rate used for training

        "cross_entropy_weight" tf.nn.weighted_cross_entropy_with_logits if != 1

        "optimizer" method used to minimize cost. Values: "GD" (gradient descent, default) or "Adam"
        """

        # tf session
        self._sess = tf.Session()

        self._numHiddenLayers = len(hiddenLayers)

        # Placeholders
        self._X = tf.placeholder(tf.float32, [None, numFeatures])
        self._yGold = tf.placeholder(tf.float32, [None, numLabels])

        # weights init
        self._initializer = tf.contrib.layers.xavier_initializer()

        self._layers = []
        # input layer
        self._input_layer = self._X
        self._layers.append(self._input_layer)
        # hidden layers (ReLU activation)
        for i in range(self._numHiddenLayers):
            self._h = tf.layers.dense(self._layers[-1], hiddenLayers[i], activation=tf.nn.relu,
                kernel_initializer=self._initializer)
            self._layers.append(self._h)
        # output layer
        self._output_layer = tf.layers.dense(self._layers[-1], numLabels, activation=None)
        self._predicted = tf.nn.sigmoid(self._output_layer)
        self._correct_pred = tf.equal(tf.round(self._predicted), self._yGold)
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_pred, tf.float32))
        # loss
        if cross_entropy_weight == 1:
            self._cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._yGold, logits=output_layer)
        else:
            self._cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                pos_weight = cross_entropy_weight,targets=self._yGold, logits=self._output_layer)
        self._cost = tf.reduce_mean(self._cross_entropy)
        if optimizer == "Adam":
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._cost)
        if optimizer == "GD":
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self._cost)
        # --->>> implement optimizer != GD or Adam handler

        # saver to save session
        self._saver = tf.train.Saver()

        return


    ####################################
    def train(
        self, numEpochs, learning_rate, trainX, trainY, 
        valX=None, valY=None, val_epochs=None
    ):
        """
        Train NN on features "trainX" an labels "trainY".

        "numEpochs" is the number of training epochs.
        
        "valX" and "valY" is the validation set.

        "val_epochs" is the number of epochs between each training evaluation 
        """

        if valX is None or valY is None:
            valX = trainX; valY = trainY
        if val_epochs is None:
            val_epochs = int(numEpochs/100)
        
        # track cost
        acc_history = np.empty(shape=[1],dtype=float)
        cost_history = np.empty(shape=[1],dtype=float)
        val_acc_history = np.empty(shape=[1],dtype=float)
        val_cost_history = np.empty(shape=[1],dtype=float)

        sess = self._sess # tf session
        sess.run(tf.global_variables_initializer())

        for step in range(numEpochs + 1):
            # train
            sess.run(self._optimizer, feed_dict={self._X: trainX, self._yGold: trainY})
            # compute cost and accuracy on TS
            loss, acc = sess.run([self._cost, self._accuracy],
                feed_dict={self._X: trainX, self._yGold: trainY})
            acc_history = np.append(acc_history, acc)
            cost_history = np.append(cost_history, loss)
            # compute cost and accuracy on VS
            if step % val_epochs == 0:
                loss, acc = sess.run([self._cost, self._accuracy],
                    feed_dict={self._X: valX, self._yGold: valY})
                val_acc_history = np.append(acc_history, acc)
                val_cost_history = np.append(cost_history, loss)
                print("TS: Epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, cost_history[-1], acc_history[-1]))
                print("VS: Epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}\n".format(step, loss, acc))

        # Final cost and accuracy on VS   
        loss, acc = sess.run([self._cost, self._accuracy],
            feed_dict={self._X: valX, self._yGold: valY})
        val_acc_history = np.append(acc_history, acc)
        val_cost_history = np.append(cost_history, loss)
        print("Final VS: Epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}\n".format(step, loss, acc))


        return acc_history, cost_history, val_acc_history, val_cost_history




    ####################################
    def predict(self, testX, testY):

        predicted, loss, acc = self._sess.run(
            [self._predicted, self._cost, self._accuracy],
            feed_dict={self._X: testX, self._yGold: testY}
        )

        print("Test Set:\tLoss: {:.3f}\tAcc: {:.2%}\n".format(loss, acc))

        return predicted, loss, acc




    ####################################
    def close_session(self):
        """
        Close tf session, release resources
        """

        self._sess.close()
        print("Tensorflow session closed.\n")

        return

    
    '''
    Error: Key beta1_power_1 not found in checkpoint

    ####################################
    def save_session(self, save_path_str):
        """
        Save tf session to file
        """

        # Save the variables to disk.
        save_path = self._saver.save(self._sess, save_path_str)
        print("Model saved in path: %s\n" % save_path)
            
        return


    
    ####################################
    def load_session(self, load_path_str):
        """
        Load tf session from file
        """

        # Load variables from disk.
        self._saver.restore(self._sess, load_path_str)
        print("Model in path: %s restored." % load_path_str)
            
        return
    '''



# Usage example
if __name__=="__main__":

    import pandas as pd # load csv data
    import numpy as np
    from sklearn.model_selection import train_test_split # split into training and test set
    import dataset as ds # load dataset

    # load dataset
    noShow = ds.import_data_df([ds._FILE_PATHS['merged']])
    noShow_X, noShow_y = noShow.iloc[:, :-1].values, noShow.iloc[:, -1].values
    noShow_y = np.array([[i] for i in noShow_y])

    # separate training and test set randomly keeping classes ratio
    trainX, testX, trainY, testY = train_test_split(
        noShow_X, noShow_y, test_size=0.33, random_state=42,
        stratify = noShow_y
    )

    # Hidden layers
    hiddenLayers = [10, 5] 
    # number of features
    numFeatures = trainX.shape[1]
    # number of classes
    numLabels = trainY.shape[1]

    # init
    NN = NN_Sigmoid(
        hiddenLayers, numFeatures, numLabels, 0.01,
        cross_entropy_weight = 4,
        optimizer = "Adam"
    )
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
    #NN.save_session("Saved_sessions/model.ckpt")
    # close tf session
    NN.close_session()

    '''
    # create new untrained NN
    NN2 = NN_Sigmoid(
        hiddenLayers, numFeatures, numLabels, 0.01,
        cross_entropy_weight = 4,
        optimizer = "Adam"
    )
    # load session
    NN2.load_session("Saved_sessions/model.ckpt")
    # test
    NN2.predict(
        testX, testY
    )
    '''
