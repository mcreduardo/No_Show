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
        self._weights = tf.placeholder(tf.float32, [None, numLabels]) # for adaboost

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
        self._cost = tf.reduce_mean(tf.math.multiply(self._cross_entropy, self._weights))
        if optimizer == "Adam":
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._cost)
        elif optimizer == "GD":
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self._cost)
        # --->>> implement optimizer != GD or Adam handler

        # compute f1 score
        self._F1_score = compute_f1_score(self._yGold, tf.round(self._predicted))

        # saver to save session
        self._saver = tf.train.Saver()

        return


    ####################################
    def train(
        self, numEpochs, trainX, trainY, 
        valX=None, valY=None, val_epochs=None, val_patience=None,
        weightsTrain=None, weightsVal=None
    ):
        """
        Train NN on features "trainX" an labels "trainY".

        "numEpochs" is the number of training epochs.
        
        "valX" and "valY" is the validation set.

        "val_epochs" is the number of epochs between each training evaluation.

        "val_patience" is the number of times that the loss on the validation 
        set can be larger than or equal to the previously smallest loss before 
        network training stops. 
        """

        if valX is None or valY is None:
            valX = trainX; valY = trainY
        if val_epochs is None:
            val_epochs = int(numEpochs/100)

        if weightsTrain is None:
            weightsTrain = (np.array(trainY).astype(int)/10)+1
        if weightsVal is None:
            weightsVal = (np.array(valY).astype(int)/10)+1
        
        # track cost
        acc_history = np.empty(shape=[1],dtype=float)
        f1_history = np.empty(shape=[1],dtype=float)
        cost_history = np.empty(shape=[1],dtype=float)
        val_acc_history = np.empty(shape=[1],dtype=float)
        val_f1_history = np.empty(shape=[1],dtype=float)
        val_cost_history = np.empty(shape=[1],dtype=float)
        val_epoch = np.empty(shape=[1],dtype=float)

        sess = self._sess # tf session
        sess.run(tf.global_variables_initializer())

        break_point = numEpochs
        for step in range(numEpochs + 1):
            # train
            sess.run(self._optimizer, feed_dict={self._X: trainX, self._yGold: trainY, self._weights: weightsTrain})
            # compute cost and accuracy on TS
            loss, acc, f1 = sess.run([self._cost, self._accuracy, self._F1_score],
                feed_dict={self._X: trainX, self._yGold: trainY, self._weights: weightsTrain})
            acc_history = np.append(acc_history, acc)
            f1_history = np.append(f1_history, f1)
            cost_history = np.append(cost_history, loss)
            # compute cost and accuracy on VS
            if step % val_epochs == 0:
                loss, acc, f1 = sess.run([self._cost, self._accuracy, self._F1_score],
                    feed_dict={self._X: valX, self._yGold: valY, self._weights: weightsVal})
                val_acc_history = np.append(val_acc_history, acc)
                val_f1_history = np.append(val_f1_history, f1)
                val_cost_history = np.append(val_cost_history, loss)
                val_epoch = np.append(val_epoch, step)
                print("TS: Epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}\tF1: {:.3f}".format(
                    step, cost_history[-1], acc_history[-1], f1_history[-1]))
                print("VS: Epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}\tF1: {:.3f}\n".format(
                    step, loss, acc, f1))
                # val patience, stop training if no improvement in loss
                if not val_patience is None: # defined by user
                    if len(val_cost_history) > val_patience + 1: # enough points for test
                        if check_val_patience(val_cost_history[1:], val_patience):
                            print("-->> Patience reached: stop training\n")
                            break_point = step
                            break # patience reached
                        
            

        # Final cost and accuracy on VS   
        loss, acc, f1 = sess.run([self._cost, self._accuracy, self._F1_score],
            feed_dict={self._X: valX, self._yGold: valY, self._weights: weightsVal})
        val_acc_history = np.append(val_acc_history, acc)
        val_f1_history = np.append(val_f1_history, f1)
        val_cost_history = np.append(val_cost_history, loss)
        val_epoch = np.append(val_epoch, break_point)
        print("Final VS: Epoch: {:5}\tLoss: {:.3f}\tAcc: {:.2%}\tF1: {:.3f}\n".format(
            step, loss, acc, f1))



        acc_history = acc_history[1:]
        f1_history = f1_history[1:]
        cost_history =cost_history[1:]
        val_acc_history = val_acc_history[1:]
        val_f1_history = val_f1_history[1:]
        val_cost_history = val_cost_history[1:]
        val_epoch = val_epoch[1:]

        return acc_history, f1_history, cost_history, val_acc_history, val_f1_history, val_cost_history, val_epoch




    ####################################
    def predict(self, testX, testY):

        weights = (np.array(testY).astype(int)/10)+1

        actual, predicted, loss, acc, f1 = self._sess.run(
            [self._yGold, self._predicted, self._cost, self._accuracy, self._F1_score],
            feed_dict={self._X: testX, self._yGold: testY, self._weights: weights}
        )

        print("Test Set:\tLoss: {:.3f}\tAcc: {:.2%}\tF1: {:.3f}\n".format(loss, acc, f1))
        
        return predicted, loss, acc, f1



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

####################################
# Helper functions
####################################

def check_val_patience(cost, val_patience):
    """
    check if training patience reached

    "cost" is vector with computed losses.

    "val_patience" is the number of times that the loss on the validation 
        set can be larger than or equal to the previously smallest loss before 
        network training stops.  

    return:
        "True" if patience limit reached
    """

    min_last_cost = min(cost[-val_patience:])
    min_cost = min(cost)
    if min_last_cost > min_cost:
        return True # patience reached
    return False


def compute_f1_score(actual, predicted):

    TP = tf.count_nonzero(predicted * actual)
    TN = tf.count_nonzero((predicted - 1) * (actual - 1))
    FP = tf.count_nonzero(predicted * (actual - 1))
    FN = tf.count_nonzero((predicted - 1) * actual)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return f1

def plot_PR_curve(actual, scores):

    return





####################################
# Usage example
####################################
if __name__=="__main__":

    import pandas as pd # load csv data
    import numpy as np
    from sklearn.model_selection import train_test_split # split into training and test set
    import dataset as ds # load dataset
    import matplotlib.pyplot as plt # plotting

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
    hiddenLayers = [10, 10, 10, 5] 
    # number of features
    numFeatures = trainX.shape[1]
    # number of classes
    numLabels = trainY.shape[1]

    # init
    NN = NN_Sigmoid(
        hiddenLayers, numFeatures, numLabels, 0.05,
        cross_entropy_weight = 4,
        optimizer = "Adam"
    )
    # train
    weights = (np.array(trainY).astype(int)/10)+.5

    acc_history, f1_history, cost_history,\
    val_acc_history, val_f1_history, val_cost_history, val_epoch = NN.train(
        1000, trainX, trainY, 
        valX=valX, valY=valY, val_epochs=25, val_patience=5,
        weightsTrain=weights
    )
    # test
    prediction, loss, acc, f1 = NN.predict(
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

   
    # Plot loss, accuracy, f1
    plt.subplot(131)
    plt.plot(range(len(cost_history)),cost_history)
    plt.plot(val_epoch,val_cost_history, 'ro-')
    xmax = max([max(range(len(cost_history))), max(val_epoch)])
    plt.hlines(loss, 0, xmax, colors='g', linestyles='solid', label='test set')
    plt.title('Cost')

    plt.subplot(132)
    plt.plot(range(len(acc_history)),acc_history)
    plt.plot(val_epoch,val_acc_history, 'ro-')
    xmax = max([max(range(len(cost_history))), max(val_epoch)])
    plt.hlines(acc, 0, xmax, colors='g', linestyles='solid', label='test set')
    plt.title('Accuracy')

    plt.subplot(133)
    plt.plot(range(len(f1_history)),f1_history)
    plt.plot(val_epoch,val_f1_history, 'ro-')
    xmax = max([max(range(len(cost_history))), max(val_epoch)])
    plt.hlines(f1, 0, xmax, colors='g', linestyles='solid', label='test set')
    plt.title('F1-Score')

    plt.show()