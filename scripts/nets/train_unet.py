"""Train u-net."""

import numpy as np
import tensorflow as tf
from unet import Unet
tf.enable_eager_execution()

# Net parameters
input_channels = 6
output_channels = 6
datatype = tf.float32

# Training parameters
RESTORE = False
epochs = 1
iters = 20
batch_size = 1
learning_rate = 1e-3
checkpoint_path = "checkpoints/unet"
checkpoint_cadence = 10

# Data loading
resolution = 16
labels = np.zeros((batch_size, resolution, resolution, resolution, output_channels))
inputs = np.zeros((batch_size, resolution, resolution, resolution, input_channels))

def load_inputs(inputs):
    inputs[:] = np.random.randn(*inputs.shape)

def load_labels(labels):
    labels[:] = np.random.randn(*labels.shape)

# Build network and optimizer
model = Unet()
optimizer = tf.train.AdamOptimizer(learning_rate)
if RESTORE:
    model.load_weights(checkpoint_path)

# Learning loop
with tf.device('/cpu:0'):
    for epoch in range(epochs):
        for iter in range(iters):
            # Loads labels and network inputs here and asign to labels and Net_input
            load_inputs(inputs)
            load_labels(labels)
            tf_inputs = tf.cast(inputs, datatype)
            tf_labels = tf.cast(labels, datatype)
            # Optimization step
            with tf.GradientTape() as tape:
                tape.watch(model.variables)
                cost, outputs = model.cost_function([tf_inputs, tf_inputs], tf_labels)
            weight_grads = tape.gradient(cost, [model.variables])
            #clipped_grads = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
            optimizer.apply_gradients(zip(weight_grads[0], model.variables), global_step=tf.train.get_or_create_global_step())
            # Status and output
            print('epoch.iter: %i.%i, Cost: %f' %(epoch, iter, cost.numpy()))
            if (iter+1) % checkpoint_cadence == 0:
                print("Saving weights.")
                model.save_weights(checkpoint_path)
