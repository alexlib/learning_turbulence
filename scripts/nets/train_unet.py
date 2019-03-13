"""Train u-net."""

import numpy as np
import tensorflow as tf
from unet import Unet
tf.enable_eager_execution()

# Parameters
RESTORE = True
train_batch_size = 1
resolution = 16
input_channels = 6
output_channels = 6
checkpoint_path = "checkpoints/model_Unet"
datatype = tf.float32

def load_inputs(Net_input):
    Net_input[:] = np.random.randn(*Net_input.shape)

def load_outputs(labels):
    labels[:] = np.random.randn(*labels.shape)

# Build network and optimizer
Unet_obj = Unet()
optimizer = tf.train.AdamOptimizer(1.0e-3)
if RESTORE:
    Unet_obj.load_weights(checkpoint_path)

# Allocate space for input and output
labels = np.zeros((train_batch_size, resolution, resolution, resolution, output_channels))
Net_input = np.zeros((train_batch_size, resolution, resolution, resolution, input_channels))

# Learning loop
with tf.device('/cpu:0'):
    for epoch in range(1):
        for train_iter in range(10):
            # Loads labels and network inputs here and asign to labels and Net_input
            load_inputs(Net_input)
            load_outputs(labels)
            tf_labels = tf.cast(labels, datatype)
            tf_Net_input = tf.cast(Net_input, datatype)
            # Optimization step
            with tf.GradientTape() as tape:
                tape.watch(Unet_obj.variables)
                cost_value, alpha_net = Unet_obj.cost_function(tf_Net_input, tf_labels)
            weight_grads = tape.gradient(cost_value, [Unet_obj.variables])
            #clipped_grads = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
            optimizer.apply_gradients(zip(weight_grads[0], Unet_obj.variables), global_step=tf.train.get_or_create_global_step())
            # Status and output
            print(epoch, train_iter, cost_value.numpy())
            if ((train_iter+1) % 10) == 0:
                print("saving weights.")
                Unet_obj.save_weights("checkpoints/model_Unet")
