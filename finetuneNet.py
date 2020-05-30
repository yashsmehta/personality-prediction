import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import KFold
import numpy as np
import csv
import pickle
import time
from datetime import timedelta

import utils


inp_dir, dataset_type, network, lr, batch_size, epochs, seed, write_file, embed, layer = utils.parse_args()
n_classes=2
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

start=time.time()

if (embed=='bert-base'):
    pretrained_weights='bert-base-uncased'
    n_hl=12
    hidden_dim=768

elif (embed=='bert-large'):
    pretrained_weights='bert-large-uncased'
    n_hl=24
    hidden_dim=1024

file = open(inp_dir+dataset_type+'-'+embed+'.pkl', 'rb')

data = pickle.load(file)
data_x, data_y = list(zip(*data))
file.close()

if(layer == 'all'):
    alphaW = np.full([n_hl], 1/n_hl)

else:
    alphaW = np.zeros([n_hl])
    alphaW[int(layer) - 1] = 1

#just changing the way data is stored (tuples of minibatches) and getting the output for the required layer of BERT using alphaW
#data_x[ii].shape = (12, batch_size, 768)
inputs = []
targets = []

n_batches = len(data_y)

for ii in range(n_batches):
    inputs.extend(np.einsum('k,kij->ij', alphaW, data_x[ii]))
    targets.extend(data_y[ii])

inputs = np.array(inputs)
targets = tf.keras.utils.to_categorical(np.array(targets), num_classes=n_classes)

n_data = targets.shape[0]

model = tf.keras.models.Sequential()

if (network  == 'fc'):
    model.add(tf.keras.layers.Dense(500, input_dim=hidden_dim, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(n_classes))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
              metrics=['mse', 'accuracy'])

print(model.summary())
validation_split = 0.15
history = model.fit(inputs, targets, epochs=epochs, batch_size=batch_size,
                    validation_split=validation_split, verbose = 1)

print('acc: ', history.history['acc'])
print('val acc: ', history.history['val_acc'])
print('loss: ', history.history['loss'])
print('val loss: ', history.history['val_loss'])

print(timedelta(seconds=int(time.time()-start)), end=' ')
# print(model.evaluate(inputs, targets, batch_size=batch_size))

if (write_file):
    results_file='results.csv'
    meta_info=(lr, epochs, seed, embed, layer)
    utils.file_writer(results_file, meta_info, acc, loss_val)