
import tensorflow as tf


# Net parameters
input_channels = 6
datatype = tf.float64
stacks = 3
stack_width = 3
filters = 6
output_channels = 6
activation = 'relu'

# Training parameters
RESTORE = False
epochs = 2
snapshots = 20
perm_seed = 978
testing_size = 2
training_size = snapshots - testing_size
learning_rate = 1e-6
checkpoint_path = "checkpoints/unet"
checkpoint_cadence = 1
diss_cost = 0
#device = "/device:GPU:0"
device = "/cpu:0"