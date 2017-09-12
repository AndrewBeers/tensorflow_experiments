import numpy as np
import tensorflow as tf

def dynamicRNN(x, seqlen, weights, biases, seq_max_len, multi_layer_network, features_schedule):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    # with tf.name_scope(RNN_NAME):

        # with tf.name_scope('Unstack'):
            # print(x.get_shape())
            x = tf.unstack(x, seq_max_len, 1)
            # print(x)

        # Define a lstm cell # with tensorflow
        # with tf.name_scope('LSTM Cell'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(output_feature_num)

        # Get lstm cell output, providing 'sequence_length' will perform dynamic
        # calculation.
        # with tf.name_scope('Static RNN'):
            outputs, states = tf.contrib.rnn.static_rnn(multi_layer_network, x, dtype=tf.float32, sequence_length=seqlen)

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
            # print(outputs.get_shape())
            outputs = tf.transpose(outputs, [1, 0, 2])
            # print(outputs.get_shape())

        # with tf.name_scope('Indexing'):
            # Hack to build the indexing and retrieve the right output.
            batch_size = tf.shape(outputs)[0]
            # Start indices for each sample
            index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
            # Indexing
            outputs = tf.gather(tf.reshape(outputs, [-1, features_schedule[-2]]), index)
            # print(outputs.get_shape())

        # with tf.name_scope('Return Function'):

        # Linear activation, using outputs computed above

            # print(weights.get_shape())
            # print(biases.get_shape())

            return_function = tf.matmul(outputs, weights) + biases

            # print(return_function.get_shape())                
            return return_function

if __name__ == '__main__':
    pass