from generate_data import ToftsSequenceData
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from qtim_tools.qtim_dce.dce_util import generate_AIF, parker_model_AIF, convert_intensity_to_concentration, revert_concentration_to_intensity, estimate_concentration

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.1
training_iters = 1000000
batch_size = 50
display_step = 10
num_layers = 3

# Network Parameters
seq_max_len = 50 # Sequence max length

trainset = ToftsSequenceData(n_samples=3000, max_seq_len=seq_max_len)
testset = ToftsSequenceData(n_samples=3000, max_seq_len=seq_max_len)

# for curve in testset.data:
    # print curve
    # plt.plot(curve)
    # plt.show()
    # # fd=gd
# print testset.data

layers = 2
features_schedule = [seq_max_len, 100, 25, 2]

weights, biases = ({}, {})
cells = []

for layer_idx in xrange(layers):
    weights['out' + str(layer_idx)] = tf.Variable(tf.random_normal([features_schedule[layer_idx+1], features_schedule[layer_idx+2]]))
    biases['out' + str(layer_idx)] = tf.Variable(tf.random_normal([features_schedule[layer_idx+2]]))
    cells.append(tf.contrib.rnn.BasicLSTMCell(features_schedule[layer_idx+1]))

multi_layer_network = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, features_schedule[-1]])
seqlen = tf.placeholder(tf.int32, [None])
layer_preds = [0]*(layers+1)
layer_preds[0] = x
# tf Graph input

def dynamicRNN(x, input_seqlen, input_feature_num, input_weights, input_biases, output_feature_num, layer_idx=0):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # with tf.name_scope(RNN_NAME):

        # with tf.name_scope('Unstack'):
            print('LAYER: ', str(layer_idx))
            print('INPUT_PARAMS: ',x,input_seqlen, input_feature_num, input_weights, input_biases, output_feature_num, layer_idx)
            print(x.get_shape())
            x = tf.unstack(x, input_feature_num, 1)
            # print(x)

        # Define a lstm cell # with tensorflow
        # with tf.name_scope('LSTM Cell'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(output_feature_num)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        # with tf.name_scope('Static RNN'):
            if layer_idx == 0:
                outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
            else:
                outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)                

        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow d oesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        # with tf.name_scope('Stack and Transpose'):
            # print(outputs)
            outputs = tf.stack(outputs)
            print(outputs.get_shape())
            outputs = tf.transpose(outputs, [1, 0, 2])
            print(outputs.get_shape())

        # with tf.name_scope('Indexing'):
            if layer_idx == 0:
                # Hack to build the indexing and retrieve the right output.
                batch_size = tf.shape(outputs)[0]
                # Start indices for each sample
                index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
                # Indexing
                outputs = tf.gather(tf.reshape(outputs, [-1, output_feature_num]), index)
                print(outputs.get_shape())

        # with tf.name_scope('Return Function'):

        # Linear activation, using outputs computed above

            if layer_idx == layers - 1:
                return_function = tf.matmul(outputs, input_weights) + input_biases

                print(input_weights.get_shape())
                print(input_biases.get_shape())

                print(return_function.get_shape())                
                return return_function
            else:
                return_function = tf.stack(tf.split(outputs, output_feature_num, axis=1))
                return_function = tf.transpose(return_function, [1,0,2])

                print(return_function.get_shape())                
                return tf.tanh(return_function)

# def dynamicRNN(x, input_seqlen, input_feature_num, weights, biases, output_feature_num, layer_idx=0):

for layer_idx in xrange(layers):

    # if layer_idx == 0:
        print layer_preds
        layer_preds[layer_idx+1] = dynamicRNN(
            x = layer_preds[layer_idx], 
            input_seqlen = seqlen, 
            input_feature_num = features_schedule[layer_idx], 
            input_weights = weights['out' + str(layer_idx)], 
            input_biases = biases['out' + str(layer_idx)],
            output_feature_num = features_schedule[layer_idx + 1], 
            layer_idx = layer_idx)

    # elif layer_idx == layers-1:
    #     pass

    # else:
    #     pass

# pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
# with tf.name_scope('Cost Function'):
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

pred = layer_preds[-1]

mse = tf.pow(pred-y, 2)
cost = tf.reduce_mean(mse)
# with tf.name_scope('Optimizer'):
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
# with tf.name_scope('Evaluate'):
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
accuracy = tf.reduce_mean(tf.abs(pred-y))

merged = tf.summary.merge_all()
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter('./tensorlog', sess.graph)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen:batch_seqlen})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            # Calculate batch loss
            # mse = sess.run(mse, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = testset.data
    test_label = testset.labels
    test_seqlen = testset.seqlen
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))
    preds = sess.run(pred, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen})
    for pred_idx, pred in enumerate(preds):
        print [round(x,3) for x in pred.tolist() + test_label[pred_idx]]
    # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))