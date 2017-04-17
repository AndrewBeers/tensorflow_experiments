from qtim_tools.qtim_dce.dce_util import generate_AIF, parker_model_AIF, convert_intensity_to_concentration, revert_concentration_to_intensity, estimate_concentration
import generate_data
import random
import numpy as np


class ToftsSequenceData(object):
    """ Generate sequence of data # with dynamic length.
    This class generate samples for training:
    - Class 0: Tofts sequences
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array # with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=50, min_seq_len=3, max_value=1000, ktrans_range=[.001,2], ve_range=[0.001,.95], gaussian_noise=[0,0], T1_range=[1000,1000], TR_range=[5, 5], flip_angle_degrees_range=[30,30], relaxivity_range=[.0045, .0045], hematocrit_range=[.45,.45], sequence_length_range=[15,45], time_interval_seconds_range=[2,2], injection_start_time_seconds_range=[10,10], T1_blood_range=[1440,1440], baseline_intensity=[100,100], outputs='joined'):
        
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
            if random.random() < .5 or True:    
                
                injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)
                time_interval_seconds = np.random.uniform(*time_interval_seconds_range)
                time_interval_minutes = time_interval_seconds/60
                scan_time_seconds = seq_len * time_interval_seconds

                while injection_start_time_seconds > .8*scan_time_seconds:
                    injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)

                AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, timepoints=seq_len)

                ktrans = np.random.uniform(*ktrans_range)
                ve = np.random.uniform(*ve_range)

                Concentration = np.array(estimate_concentration([ktrans, ve], AIF, time_interval_minutes))

                Intensity = revert_concentration_to_intensity(data_numpy=Concentration, reference_data_numpy=[], T1_tissue=np.random.uniform(*T1_range), TR=np.random.uniform(*TR_range), flip_angle_degrees=np.random.uniform(*flip_angle_degrees_range), injection_start_time_seconds=injection_start_time_seconds, relaxivity=np.random.uniform(*relaxivity_range), time_interval_seconds=time_interval_seconds, hematocrit=np.random.uniform(*hematocrit_range), T1_blood=0, T1_map = [], static_baseline=np.random.uniform(*baseline_intensity)).tolist()

                Intensity = Intensity - np.mean(Intensity) / np.std(Intensity)

                s = []
                s += [[value] for value in Intensity]
                s += [[0.] for i in range(max_seq_len - seq_len)]

                self.data.append(s)
                self.labels.append([ktrans, ve])        

            # else:
            #     # Generate a random sequence
            #     s = [[np.random.uniform(baseline_intensity[0]*2, baseline_intensity[0]*3)]
            #          for i in range(seq_len)]
            #     # Pad sequence for dimension consistency
            #     s += [[0.] for i in range(max_seq_len - seq_len)]

            #     self.data.append(s)
            #     self.labels.append([0., 1.])

            # else:
                
            #     injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)
            #     time_interval_seconds = np.random.uniform(*time_interval_seconds_range)
            #     time_interval_minutes = time_interval_seconds/60
            #     scan_time_seconds = seq_len * time_interval_seconds

            #     while injection_start_time_seconds > .8*scan_time_seconds:
            #         injection_start_time_seconds = np.random.uniform(*injection_start_time_seconds_range)

            #     AIF = parker_model_AIF(scan_time_seconds, injection_start_time_seconds, time_interval_seconds, timepoints=seq_len)

            #     Concentration = np.array(estimate_concentration([np.random.uniform(*ktrans_low_range),np.random.uniform(*ve_range)], AIF, time_interval_minutes))

            #     Intensity = revert_concentration_to_intensity(data_numpy=Concentration, reference_data_numpy=[], T1_tissue=np.random.uniform(*T1_range), TR=np.random.uniform(*TR_range), flip_angle_degrees=np.random.uniform(*flip_angle_degrees_range), injection_start_time_seconds=injection_start_time_seconds, relaxivity=np.random.uniform(*relaxivity_range), time_interval_seconds=time_interval_seconds, hematocrit=np.random.uniform(*hematocrit_range), T1_blood=0, T1_map = [], static_baseline=np.random.uniform(*baseline_intensity)).tolist()

            #     s = []

            #     Intensity = Intensity - np.mean(Intensity) / np.std(Intensity)

            #     s += [[value] for value in Intensity]
            #     s += [[0.] for i in range(max_seq_len - seq_len)]

            #     self.data.append(s)
            #     self.labels.append([0., 1.])        


        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))


        return batch_data, batch_labels, batch_seqlen
