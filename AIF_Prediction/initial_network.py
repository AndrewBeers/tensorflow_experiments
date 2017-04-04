import generate_data
import numpy as np

Generated_AIF_Intensities = np.load('Generated_AIF_Intensities.npy')
Groundtruth_AIF_Intensities = np.load('Groundtruth_AIF_Intensities.npy')

Generated_AIF_Intensities = np.reshape(Generated_AIF_Intensities, (Generated_AIF_Intensities[0]* Generated_AIF_Intensities[1], Generated_AIF_Intensities[2]))
Groundtruth_AIF_Intensities = np.reshape(Groundtruth_AIF_Intensities, (Groundtruth_AIF_Intensities[0]* Groundtruth_AIF_Intensities[1], Groundtruth_AIF_Intensities[2]))

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
    def __init__(self, n_samples=1000, max_seq_len=1000, min_seq_len=3, max_value=1000, gaussian_noise=0):
        self.data = []
        self.labels = []
        self.seqlen = []
        for i in range(n_samples):
            # Random sequence length
            len = random.randint(min_seq_len, max_seq_len)
            # Monitor sequence length for TensorFlow dynamic calculation
            self.seqlen.append(len)
            # Add a random or linear int sequence (50% prob)
            if random.random() < .5:
                # Generate a linear sequence
                s = [[float(i)/max_value] for i in
                     range(rand_start, rand_start + len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len(s))]
                self.data.append(s)
                self.labels.append([1., 0.])
            else:
                # Generate a random sequence
                s = [[float(random.randint(0, max_value))/max_value]
                     for i in range(len)]
                # Pad sequence for dimension consistency
                s += [[0.] for i in range(max_seq_len - len)]
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