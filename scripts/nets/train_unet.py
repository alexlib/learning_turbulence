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
snapshots = 200
perm_seed = 978
testing_size = 20
training_size = 20
learning_rate = 1e-3
checkpoint_path = "checkpoints/unet"
checkpoint_cadence = 10
diss_cost = 0
device = "/device:GPU:0"
#device = "/cpu:0"

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
model = unet.Unet(stacks, stack_width, filters, output_channels, activation=activation)
optimizer = tf.train.AdamOptimizer(learning_rate)
if RESTORE:
    model.load_weights(checkpoint_path)

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
    tens_array = np.array([[c['xx'], c['xy'], c['xz']],
                           [c['yx'], c['yy'], c['yz']],
                           [c['zx'], c['zy'], c['zz']]])
    return tens_array

def deviatoric_part(tens):
    """Compute deviatoric part of tensor."""
    tr_tens = np.trace(tens)
    tens_d = tens.copy()
    N = tens.shape[0]
    for i in range(N):
        tens_d[i, i] = tens[i, i] - tr_tens / N
    return tens_d

def cost_function(inputs, labels):
    # Call net
    outputs = model.call(inputs)
    # Load components into object arrays, take deviatoric part of stresses
    S_true = array_of_tf_components(inputs[0])
    tau_d_true = deviatoric_part(array_of_tf_components(labels))
    tau_d_pred = deviatoric_part(array_of_tf_components(outputs))
    # Pointwise deviatoric stress error
    tau_d_diff = tau_d_pred - f_tau_d_true
    f2_tau_d_diff = np.trace(np.dot(tau_d_diff, tau_d_diff.T))
    f2_tau_d_true = np.trace(np.dot(tau_d_true, tau_d_true.T))
    # Normalized L2-squared deviatoric stress error
    L2_tau_d_error = tf.reduce_mean(f2_tau_d_diff) / tf.reduce_mean(f2_tau_d_true)
    # Pointwise dissipation error
    D_true = np.trace(np.dot(tau_d_true, S_true.T))
    D_pred = np.trace(np.dot(tau_d_pred, S_true.T))
    D_diff = D_true - D_pred
    # Normalized L2-squared dissipation error
    L2_D_error = tf.reduce_mean(D_diff**2) / tf.reduce_mean(D_true**2)
    return (1-diss_cost) * L2_tau_d_error + diss_cost * L2_D_error

# Learning loop
costs = []
with tf.device(device):
    for epoch in range(epochs):
        print(f"Beginning epoch {epoch}")
        costs_epoch = []
        for savenum in rand.permutation(snapshots_train):
            # Load adjascent outputs
            inputs_0, labels_0 = load_data(savenum)
            #inputs_1, labels_1 = load_data(savenum+1)
            # Combine inputs to predict later output
            #tf_inputs = [tf.cast(inputs_0, datatype), tf.cast(inputs_1, datatype)]
            #tf_labels = tf.cast(labels_1, datatype)
            tf_inputs = [tf.cast(inputs_0, datatype)]
            tf_labels = tf.cast(labels_0, datatype)
            # Run optimization
            with tf.GradientTape() as tape:
                tape.watch(model.variables)
                cost, tf_outputs = cost_function(tf_inputs, tf_labels)
            weight_grads = tape.gradient(cost, model.variables)
            optimizer.apply_gradients(zip(weight_grads, model.variables), global_step=tf.train.get_or_create_global_step())
            # Status and output
            costs_epoch.append(cost.numpy())
            print('epoch.iter: %i.%i, cost: %f' %(epoch, iter, cost.numpy()))
            if (iter+1) % checkpoint_cadence == 0:
                print("Saving weights.")
                model.save_weights(checkpoint_path)
        costs.append(costs_epoch)
costs = np.array(costs)
np.save('costs.npy', costs)
