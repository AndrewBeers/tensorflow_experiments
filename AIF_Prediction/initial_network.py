import generate_data
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from qtim_tools.qtim_dce.dce_util import generate_AIF, parker_model_AIF, convert_intensity_to_concentration, revert_concentration_to_intensity, estimate_concentration

# Generated_AIF_Intensities = np.load('Generated_AIF_Intensities.npy')
# Groundtruth_AIF_Intensities = np.load('Groundtruth_AIF_Intensities.npy')

# Generated_AIF_Intensities = np.reshape(Generated_AIF_Intensities, (Generated_AIF_Intensities[0]* Generated_AIF_Intensities[1], Generated_AIF_Intensities[2]))
# Groundtruth_AIF_Intensities = np.reshape(Groundtruth_AIF_Intensities, (Groundtruth_AIF_Intensities[0]* Groundtruth_AIF_Intensities[1], Groundtruth_AIF_Intensities[2]))

SEQ_MAX_LEN = 200

# ====================
#  TOFTS DATA GENERATOR
# ====================
class ToftsSequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: Tofts sequences
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=SEQ_MAX_LEN, min_seq_len=3, max_value=1000, ktrans_range=[.3,2], ve_range=[0.001,.95], gaussian_noise=[0,0], T1_range=[1000,1000], TR_range=[5, 5], flip_angle_degrees_range=[30,30], relaxivity_range=[.0045, .0045], hematocrit_range=[.45,.45], sequence_length_range=[70,140], time_interval_seconds_range=[2,2], injection_start_time_seconds_range=[10,10], T1_blood_range=[1440,1440], baseline_intensity=[100,100]):
        
        ktrans_low_range = [.001, .3]

        self.data = []
        self.labels = []
        self.seqlen = []

        for i in range(n_samples):

            # Random sequence length
            seq_len = np.random.random_integers(*sequence_length_range)
            
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(seq_len)
            
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:    
                
                injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)
                time_interval_seconds = np.random.uniform(*time_interval_seconds_range)
                time_interval_minutes = time_interval_seconds/60
                scan_time_seconds = seq_len * time_interval_seconds

                while injection_start_time_seconds > .8*scan_time_seconds:
                    injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)

                AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, timepoints=seq_len)

                Concentration = np.array(estimate_concentration([np.random.uniform(*ktrans_range),np.random.uniform(*ve_range)], AIF, time_interval_minutes))

                Intensity = revert_concentration_to_intensity(data_numpy=Concentration, reference_data_numpy=[], T1_tissue=np.random.uniform(*T1_range), TR=np.random.uniform(*TR_range), flip_angle_degrees=np.random.uniform(*flip_angle_degrees_range), injection_start_time_seconds=injection_start_time_seconds, relaxivity=np.random.uniform(*relaxivity_range), time_interval_seconds=time_interval_seconds, hematocrit=np.random.uniform(*hematocrit_range), T1_blood=0, T1_map = [], static_baseline=np.random.uniform(*baseline_intensity)).tolist()

                Intensity = Intensity - np.mean(Intensity) / np.std(Intensity)

                s = []
                s += [[value] for value in Intensity]
                s += [[0.] for i in range(max_seq_len - seq_len)]

                self.data.append(s)
                self.labels.append([1., 0.])        

            # else:
            #     # Generate a random sequence
            #     s = [[np.random.uniform(baseline_intensity[0]*2, baseline_intensity[0]*3)]
            #          for i in range(seq_len)]
            #     # Pad sequence for dimension consistency
            #     s += [[0.] for i in range(max_seq_len - seq_len)]

            #     self.data.append(s)
            #     self.labels.append([0., 1.])

            else:
                
                injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)
                time_interval_seconds = np.random.uniform(*time_interval_seconds_range)
                time_interval_minutes = time_interval_seconds/60
                scan_time_seconds = seq_len * time_interval_seconds

                while injection_start_time_seconds > .8*scan_time_seconds:
                    injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)

                AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, timepoints=seq_len)

                Concentration = np.array(estimate_concentration([np.random.uniform(*ktrans_low_range),np.random.uniform(*ve_range)], AIF, time_interval_minutes))

                Intensity = revert_concentration_to_intensity(data_numpy=Concentration, reference_data_numpy=[], T1_tissue=np.random.uniform(*T1_range), TR=np.random.uniform(*TR_range), flip_angle_degrees=np.random.uniform(*flip_angle_degrees_range), injection_start_time_seconds=injection_start_time_seconds, relaxivity=np.random.uniform(*relaxivity_range), time_interval_seconds=time_interval_seconds, hematocrit=np.random.uniform(*hematocrit_range), T1_blood=0, T1_map = [], static_baseline=np.random.uniform(*baseline_intensity)).tolist()

                s = []

                Intensity = Intensity - np.mean(Intensity) / np.std(Intensity)

                s += [[value] for value in Intensity]
                s += [[0.] for i in range(max_seq_len - seq_len)]

                self.data.append(s)
                self.labels.append([0., 1.])        


        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))


        return batch_data, batch_labels, batch_seqlen


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_iters = 1000000
batch_size = 300
display_step = 10

# Network Parameters
seq_max_len = SEQ_MAX_LEN # Sequence max length
n_hidden = 600 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = ToftsSequenceData(n_samples=3000, max_seq_len=seq_max_len)
testset = ToftsSequenceData(n_samples=1500, max_seq_len=seq_max_len)

# for curve in testset.data:
#     # print curve
#     plt.plot(curve)
#     plt.show()
    # fd=gd
# print testset.data


# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 1])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
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
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))