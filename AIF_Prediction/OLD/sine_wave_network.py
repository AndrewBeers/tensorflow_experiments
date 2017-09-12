import numpy as np 
import random
import tensorflow as tf
import math
import matplotlib.pyplot as plt

# from generate_data import Generate_Sine_Data
from tensorflow.contrib.rnn.python.ops import core_rnn

def load_data(data_generator, n_samples,ratio):

    data_object = data_generator
    # data_object = data_generator(n_samples=n_samples, amplitude_range=[1,5], period_range=[.1,2], x_shift_range=[1,1], y_shift_range=[1,1])

    # print data_object.data

    X = np.array(data_object.data)
    y = np.array(data_object.labels)
    N = X.shape[0]

    ratio = (ratio*N).astype(np.int32)
    ind = np.random.permutation(N)

    X_train = X[ind[:ratio[0]],:]
    X_val = X[ind[ratio[0]:ratio[1]],:]
    X_test = X[ind[ratio[1]:],:]

    y_train = y[ind[:ratio[0]]]
    y_val = y[ind[ratio[0]:ratio[1]]]
    y_test = y[ind[ratio[1]:]]

    return X_train,X_val,X_test,y_train,y_val,y_test


def sample_batch(X_train,y_train,batch_size):
  """ Function to sample a batch for training"""
  N,data_len = X_train.shape
  ind_N = np.random.choice(N,batch_size,replace=False)
  X_batch = X_train[ind_N]
  y_batch = y_train[ind_N]
  return X_batch,y_batch

class Sine_Model():

    def __init__(self, num_layers, hidden_features, max_grad_norm, batch_size, sl, learning_rate, num_classes):

        self.data_input = tf.placeholder(tf.float32, [None, sl], name = 'input')
        self.data_labels = tf.placeholder(tf.float32, [None, num_classes], name='labels')
        self.dropout_probability = tf.placeholder("float", name ="Dropout_Keep_Probability")

        with tf.name_scope("LSTM_Setup") as scope:

            def single_cell():
                return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(hidden_features), output_keep_prob=self.dropout_probability)

            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for x in range(num_layers)])
            initial_state = cell.zero_state(batch_size, tf.float32)

        input_list = tf.unstack(tf.expand_dims(self.data_input, axis=2), axis=1)

        # print input_list.get_shape()

        outputs, _ = core_rnn.static_rnn(cell, input_list, dtype=tf.float32)

        self.output = outputs[-1]

        with tf.name_scope("Softmax") as scope:
            with tf.variable_scope("Softmax_params"):
                softmax_w = tf.get_variable("softmax_w", [hidden_features, num_classes])
                softmax_b = tf.get_variable("softmax_b", [num_classes])
            self.logits = tf.nn.xw_plus_b(self.output, softmax_w, softmax_b)
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.data_labels, name="softmax")
            loss = tf.pow(self.logits - self.data_labels, 2)
            self.cost = tf.reduce_mean(loss)

        with tf.name_scope("Evaluating_self.accuracy") as scope:
            # self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.data_labels)
            # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
            self.accuracy = tf.reduce_mean(loss)
            h1 = tf.summary.scalar('self.accuracy', self.accuracy)
            h2 = tf.summary.scalar('self.cost', self.cost)

        with tf.name_scope("Optimizer") as scope:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), max_grad_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = zip(grads, tvars)
            self.train_op = optimizer.apply_gradients(gradients)

            self.merged = tf.summary.merge_all()
            self.init_op = tf.global_variables_initializer()
            print('FINISHED GRAPH')

def Estimate_Sine_Generator(n_samples_train_test=[200,200]):

    trainset = Generate_Sine_Data(n_samples=n_samples_train_test[0], period_range=[.1, 1])
    testset = Generate_Sine_Data(n_samples=n_samples_train_test[1], period_range=[.1, 1])

    # Visualize Dataset
    for curve in testset.data:
        print curve
        plt.plot(curve)
        plt.show()

if __name__ == '__main__':

    Estimate_Sine_Generator()
