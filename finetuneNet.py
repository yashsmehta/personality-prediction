import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.model_selection import KFold
import numpy as np
import csv
import pickle
import time

import utils


n_classes=2
inp_dir, dataset_type, network, lr, batch_size, epochs, seed, write_file, embed, layer = utils.parse_args()

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
n_data = len(targets)

for ii in range(n_data):
    inputs.extend(np.einsum('k,kij->ij', alphaW, data_x[ii]))
    targets.extend(data_y[ii])

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
steps_per_epoch = int(((1-validation_split)*n_data)/batch_size)
validation_steps = int((validation_split*n_data)/batch_size)

history = model.fit(inputs, targets, epochs=epochs, batch_size=batch_size,
                    validation_split=validation_split, verbose = 1, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

elapsed_time = time.time() - start_time

print(history.history['accuracy'])
print(history.history['loss'])

# print(model.evaluate(inputs, targets, batch_size=1000))

# if (write_file):
#     results_file='results.csv'
#     meta_info=(lr, epochs, seed, embed, layer)
#     utils.file_writer(results_file, meta_info, acc, loss_val)
# # hidden_features || targets : (n_hl, 32, hidden_dim) || (32, 2)