"""Train u-net."""

import numpy as np
import xarray
import tensorflow as tf
import unet
import time
tf.enable_eager_execution()
from parameters import *


# Divide training data
# Randomly permute snapshots
rand = np.random.RandomState(seed=perm_seed)
snapshots_perm = 1 + rand.permutation(snapshots)
# Select testing data
snapshots_test = snapshots_perm[:testing_size]
# Select training data
snapshots_train = snapshots_perm[testing_size:testing_size+training_size]

# Data loading
def load_data(savenum):
    filename = 'filtered/snapshots_s%i.nc' %savenum
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
tf.set_random_seed(tf_seed)
model = unet.Unet(stacks, stack_width, filters_base, output_channels, **unet_kw)
optimizer = tf.train.AdamOptimizer(learning_rate)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, net=model)
if restore_epoch:
    restore_path = f"{checkpoint_path}-{restore_epoch}"
    checkpoint.restore(restore_path)#.assert_consumed()
    print('Restored from {}'.format(restore_path))
else:
    print('Initializing from scratch.')
initial_epoch = checkpoint.save_counter.numpy() + 1

# Define cost function
def array_of_tf_components(tf_tens):
    """Create object array of tensorflow packed tensor components."""
    # Collect components
    # Tensorflow shaped as (batch, *shape, channels)
    comps = ['xx', 'yy', 'zz', 'xy', 'yz', 'zx']
    c = {comp: tf_tens[..., n] for n, comp in enumerate(comps)}
    c['yx'] = c['xy']
    c['zy'] = c['yz']
    c['xz'] = c['zx']
    # Build object array
    tens_array = np.array([[None, None, None],
                           [None, None, None],
                           [None, None, None]], dtype=object)
    for i, si in enumerate(['x', 'y', 'z']):
        for j, sj in enumerate(['x', 'y', 'z']):
            tens_array[i, j] = c[si+sj]
    return tens_array

def deviatoric_part(tens):
    """Compute deviatoric part of tensor."""
    tr_tens = np.trace(tens)
    tens_d = tens.copy()
    N = tens.shape[0]
    for i in range(N):
        tens_d[i, i] = tens[i, i] - tr_tens / N
    return tens_d

def cost_function(inputs, outputs, labels):
    # Load components into object arrays, take deviatoric part of stresses
    S_true = array_of_tf_components(inputs[0])
    tau_d_true = deviatoric_part(array_of_tf_components(labels))
    tau_d_pred = deviatoric_part(array_of_tf_components(outputs))
    # Pointwise deviatoric stress error
    tau_d_diff = tau_d_pred - tau_d_true
    f2_tau_d_diff = np.trace(np.dot(tau_d_diff, tau_d_diff.T))
    # L2-squared deviatoric stress error
    L2_tau_d_error = tf.reduce_mean(f2_tau_d_diff)
    # Pointwise dissipation error
    D_true = np.trace(np.dot(tau_d_true, S_true.T))
    D_pred = np.trace(np.dot(tau_d_pred, S_true.T))
    D_diff = D_true - D_pred
    # L2-squared dissipation error
    L2_D_error = tf.reduce_mean(D_diff**2)
    cost = (1-diss_cost) * L2_tau_d_error + diss_cost * L2_D_error
    return cost

# Learning loop
training_costs = []
testing_costs = []
#with tf.device(device):
for epoch in range(initial_epoch, initial_epoch + epochs):
    print(f"Beginning epoch {epoch}", flush=True)
    # Train
    costs_epoch = []
    rand.seed(perm_seed + epoch)
    for iteration, savenum in enumerate(rand.permutation(snapshots_train)):
        # Load adjascent outputs
        inputs_0, labels_0 = load_data(savenum)
        #inputs_1, labels_1 = load_data(savenum+1)
        with tf.device(device):
            # Combine inputs to predict later output
            #tf_inputs = [tf.cast(inputs_0, datatype), tf.cast(inputs_1, datatype)]
            #tf_labels = tf.cast(labels_1, datatype)
            tf_inputs = [tf.cast(inputs_0, datatype)]
            tf_labels = tf.cast(labels_0, datatype)
            # Run optimization
            with tf.GradientTape() as tape:
                tape.watch(model.variables)
                tf_outputs = model.call(tf_inputs)
                cost = cost_function(tf_inputs, tf_outputs, tf_labels)
            weight_grads = tape.gradient(cost, model.variables)
            optimizer.apply_gradients(zip(weight_grads, model.variables), global_step=tf.train.get_or_create_global_step())
        # Status and output
        costs_epoch.append(cost.numpy())
        print('epoch.iter.save: %i.%i.%i, training cost: %.3e' %(epoch, iteration, savenum, cost.numpy()), flush=True)
    print("Saving weights.", flush=True)
    checkpoint.save(checkpoint_path)
    training_costs.append(costs_epoch)
    # Test
    costs_epoch = []
    for iteration, savenum in enumerate(snapshots_test):
        # Load adjascent outputs
        inputs_0, labels_0 = load_data(savenum)
        #inputs_1, labels_1 = load_data(savenum+1)
        with tf.device(device):
            # Combine inputs to predict later output
            #tf_inputs = [tf.cast(inputs_0, datatype), tf.cast(inputs_1, datatype)]
            #tf_labels = tf.cast(labels_1, datatype)
            tf_inputs = [tf.cast(inputs_0, datatype)]
            tf_labels = tf.cast(labels_0, datatype)
            tf_outputs = model.call(tf_inputs)
            cost = cost_function(tf_inputs, tf_outputs, tf_labels)
        # Status and output
        costs_epoch.append(cost.numpy())
        print('epoch.iter.save: %i.%i.%i, testing cost: %.3e' %(epoch, iteration, savenum, cost.numpy()), flush=True)
    testing_costs.append(costs_epoch)
training_costs = np.array(training_costs)
testing_costs = np.array(testing_costs)
np.save('training_costs.npy', training_costs)
np.save('testing_costs.npy', testing_costs)
