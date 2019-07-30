
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
restore_epoch = 0
epochs = 10
snapshots = 2000
testing_size = 200
training_size = snapshots - testing_size
perm_seed = 978
tf_seed = 718
learning_rate = 1e-3
checkpoint_path = "checkpoints/unet"
diss_cost = 0
device = "/device:GPU:0"
#device = "/cpu:0"
