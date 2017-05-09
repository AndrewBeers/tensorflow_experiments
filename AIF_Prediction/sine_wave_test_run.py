"""
LSTM for time series classification
This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
from sine_wave_network import Sine_Model,sample_batch,load_data
from generate_data import ToftsSequenceData, Generate_Sine_Data
import os
import glob
import csv

for eventfile in glob.glob('./tensorlog/*tfevents*'):
  os.remove(eventfile)

"""Load the data"""
n_samples = 8000
ratio = np.array([0.5, 0.6]) #Ratios where to split the training and validation set
# X_train,X_val,X_test,y_train,y_val,y_test = load_data(Generate_Sine_Data(n_samples=n_samples, amplitude_range=[1,5], period_range=[.1,2], x_shift_range=[1,1], y_shift_range=[1,1]),n_samples,ratio)

X_train,X_val,X_test,y_train,y_val,y_test = load_data(ToftsSequenceData(n_samples=n_samples),n_samples,ratio)

# for curve in X_train:
#     print curve
#     plt.plot(curve)
#     plt.show()

N,sl = X_train.shape
num_classes = len(np.unique(y_train))

num_classes = 1

"""Hyperparamaters"""
batch_size = 100
max_iterations = 3000
dropout = 0.8
num_layers = 3          #number of layers of stacked RNN's
hidden_features = 120             #memory cells in a layer
max_grad_norm = 5           #maximum gradient norm during training
learning_rate = .005
display_step = 2

epochs = np.floor(batch_size*max_iterations / N)
print('Train %.0f samples in approximately %d epochs' %(N,epochs))


# print X_train.shape
# print y_train.shape
# print X_val.shape

#Instantiate a model
model = Sine_Model(num_layers, hidden_features, max_grad_norm, batch_size, sl, learning_rate, num_classes)

"""Session time"""
sess = tf.Session() #Depending on your use, do not forget to close the session
writer = tf.summary.FileWriter('./tensorlog', sess.graph)  #writer for Tensorboard
sess.run(model.init_op)

cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
acc_train_ma = 0.0
try:
  for i in range(max_iterations):
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size)

    #Next line does the actual training
    output, cost_train, acc_train, _ = sess.run([model.logits, model.cost,model.accuracy, model.train_op],feed_dict = {model.data_input: X_batch,model.data_labels: y_batch,model.dropout_probability:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
    if i%display_step == 1:
    #Evaluate validation performance
      X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
      cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.data_input: X_batch, model.data_labels: y_batch, model.dropout_probability:1.0})
      print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
      # print output.shape, y_batch.shape
      # for x in xrange(output.shape[0]-5, output.shape[0]):
        # print('Output', output[x], y_batch[x])
      # print('Output', output[0], 'Label', y_batch[0])
      #Write information to .TensorBoard
      writer.add_summary(summ, i)
      writer.flush()
except KeyboardInterrupt:
  pass

X_batch, y_batch = X_test, y_test

    #Next line does the actual training
preds = sess.run(model.logits,feed_dict = {model.data_input: X_batch,model.data_labels: y_batch,model.dropout_probability:dropout})

output_filename = './results/sinetest.csv'

with open(output_filename, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    for pred_idx, pred in enumerate(preds):
        prediction = [str(pred[0]), str(y_batch[pred_idx,0])]
        print(prediction)
        writer.writerow(prediction)

epoch = float(i)*batch_size/N
print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_val,cost_val))

#now run in your terminal:
# $ tensorboard --logdir = <summaries_dir>
# Replace <summaries_dir> with your own dir