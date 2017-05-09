import numpy as np
import tensorflow as tf

from sheri_network import AIF_Network

def sheri_search(gpu=-1):

    for layer_1 in np.arange(10,110,10):
        for layer_2 in np.arange(10, 110, 10):
            for batch_size in [25,50,75,100]:
                AIF_Network(training_iters=50000, batch_size=batch_size, layers=3, features_schedule=[50,layer_1,layer_2,2], output_filename='shortrun' + str(layer_1) + '_' + str(layer_2) + '_batchsize_' + str(batch_size) + '.csv', gpu=-1)

    return

if __name__ == '__main__':
    sheri_search()