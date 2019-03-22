"""Train u-net."""

import numpy as np
import xarray
import tensorflow as tf
import unet
tf.enable_eager_execution()

# Net parameters
input_channels = 6
datatype = tf.float32
stacks = 3
stack_width = 3
filters = 6
output_channels = 6
activation = 'relu'

# Training parameters
RESTORE = False
epochs = 1
iters = 20
batch_size = 1
learning_rate = 1e-3
checkpoint_path = "checkpoints/unet"
checkpoint_cadence = 10

# Data loading
def load_data(savenum):
    filename = 'snapshots_s%i.nc' %savenum
    dataset = xarray.open_dataset(filename)
    comps = ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']
    # Strain rate
    S = [dataset['S'+c].data for c in comps]
    # Subgrid stress
    tau = [dataset['t'+c].data for c in comps]
    # Deviatoric subgrid stress
    tr_tau = tau[0] + tau[1] + tau[2]
    tau[0] = tau[0] - tr_tau/3
    tau[1] = tau[1] - tr_tau/3
    tau[2] = tau[2] - tr_tau/3
    # Reshape as (batch, *shape, channels)
    inputs = np.moveaxis(np.array(S), 0, -1)[None]
    labels = np.moveaxis(np.array(tau), 0, -1)[None]
    return inputs, labels

# Build network and optimizer
model = unet.Unet(stacks, stack_width, filters, output_channels, activation=activation)
optimizer = tf.train.AdamOptimizer(learning_rate)
if RESTORE:
    model.load_weights(checkpoint_path)

# Learning loop
with tf.device('/cpu:0'):
    for epoch in range(epochs):
        for iter in range(iters):
            savenum = iter  ## CHANGE
            # Load adjascent outputs
            inputs_0, labels_0 = load_data(savenum)
            inputs_1, labels_1 = load_data(savenum+1)
            # Combine inputs to predict later output
            tf_inputs = [tf.cast(inputs_0, datatype), tf.cast(inputs_1, datatype)]
            tf_labels = tf.cast(labels_1, datatype)
            # Run optimization
            with tf.GradientTape() as tape:
                tape.watch(model.variables)
                cost, tf_outputs = model.cost_function(tf_inputs, tf_labels)
            weight_grads = tape.gradient(cost, model.variables)
            optimizer.apply_gradients(zip(weight_grads, model.variables), global_step=tf.train.get_or_create_global_step())
            # Status and output
            print('epoch.iter: %i.%i, cost: %f' %(epoch, iter, cost.numpy()))
            if (iter+1) % checkpoint_cadence == 0:
                print("Saving weights.")
                model.save_weights(checkpoint_path)
