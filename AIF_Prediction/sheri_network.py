import csv
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from generate_data import ToftsSequenceData
from networks_repo import dynamicRNN

def AIF_Network(learning_rate = 0.1, training_iters = 1000000, batch_size = 50, display_step = 10, layers = 3, features_schedule = [50, 10, 100, 2], seq_max_len= 50, n_samples_train_test=[3000, 3000], output_filename = 'AIF_Network_Predictions.csv', gpu=0):

    # config = tf.ConfigProto(tf.GPUOptions(visible_device_list=gpu))
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu

    trainset = ToftsSequenceData(n_samples=n_samples_train_test[0], max_seq_len=seq_max_len)
    testset = ToftsSequenceData(n_samples=n_samples_train_test[1], max_seq_len=seq_max_len)

    # for curve in testset.data:
        # print curve
        # plt.plot(curve)
        # plt.show()
        # # fd=gd
    # print testset.data

    assert features_schedule[0] == seq_max_len, 'First entry in features schedule must be the maximum length of the sequence'
    assert len(features_schedule) == layers + 1, 'Features schedule must have one more entry than the number of layers.'

    weights, biases = ({}, {})
    cells = []

    weights['out'] = tf.Variable(tf.random_normal([features_schedule[-2], features_schedule[-1]]))
    biases['out'] = tf.Variable(tf.random_normal([features_schedule[-1]]))

    for layer_idx in xrange(layers):
        cells.append(tf.contrib.rnn.BasicLSTMCell(features_schedule[layer_idx+1]))

    multi_layer_network = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

    x = tf.placeholder("float", [None, seq_max_len, 1])
    y = tf.placeholder("float", [None, features_schedule[-1]])
    seqlen = tf.placeholder(tf.int32, [None])
    layer_preds = [0]*(layers+1)
    layer_preds[0] = x

    pred = dynamicRNN(x, seqlen, weights['out'], biases['out'], seq_max_len, multi_layer_network, features_schedule)

    # Define loss and optimizer
    # with tf.name_scope('Cost Function'):

    # For classification problems..
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

    mse = tf.pow(pred-y, 2)
    cost = tf.reduce_mean(mse)
    # with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    # with tf.name_scope('Evaluate'):
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

    # For classifciation problems..
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    accuracy = tf.reduce_mean(tf.abs(pred-y))

    # TODO Look up tensorboard commands.
    # accuracy_summary = tf.summary.scalar("training_accuracy", accuracy)
    # prediction_summary = tf.summary.scalar("cost", cost)
    # summary_op = tf.merge_all_summaries()

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:

        sess.run(init)

        train_writer = tf.summary.FileWriter('./tensorlog', sess.graph)

        step = 1
        while step * batch_size < training_iters:
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen:batch_seqlen})

            if step % display_step == 0:

                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
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

        preds = sess.run(pred, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen})

        with open(output_filename, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            for pred_idx, pred in enumerate(preds):
                prediction = [round(x,3) for x in pred.tolist() + test_label[pred_idx]]
                print(prediction)
                writer.writerow(prediction)

if __name__ == '__main__':
    AIF_Network(training_iters=200000, batch_size=75, layers=3, features_schedule=[50,30,10,2], output_filename='shortrun.csv', gpu="1")