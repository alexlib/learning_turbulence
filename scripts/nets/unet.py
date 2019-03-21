"""Define u-net."""

"""
Issues:
    - Why did the upconvolutions have a different kernel size?
"""

import tensorflow as tf


def axslice(axis, start, stop, step=None):
    """Slice array along a specified axis."""
    if axis < 0:
        raise ValueError("`axis` must be positive")
    slicelist = [slice(None)] * axis
    slicelist.append(slice(start, stop, step))
    return tuple(slicelist)


def pad_axis_periodic(tensor, axis, pad_left, pad_right):
    """Periodically pad tensor along a single axis."""
    N = tensor.shape[axis]
    left = tensor[axslice(axis, N-pad_left, N)]
    right = tensor[axslice(axis, 0, pad_right)]
    return tf.concat([left, tensor, right], axis)


def compress_axis_periodic(tensor, axis, pad_left, pad_right):
    """Additively compress periodic padding along a single axis."""
    N_ext = tensor.shape[axis]
    # Take center of tensor
    center = tensor[axslice(axis, pad_left, N_ext-pad_right)]
    # Build tensor of the periodic image
    left = tensor[axslice(axis, 0, pad_left)]
    right = tensor[axslice(axis, N_ext-pad_right, N_ext)]
    void_shape = tensor.shape.as_list()
    void_shape[axis] = N_ext - 2*pad_left - 2*pad_right
    void = tf.zeros(void_shape, dtype=tensor.dtype)
    image = tf.concat([right, void, left], axis)
    return center + image
    # Can't add via slices because tensor objects don't support assigment in eager mode
    #center[axslice(axis, 0, pad_right)] += tensor[axslice(axis, N_ext-pad_right, N_ext)]
    #center[axslice(axis, N_ext-pad_left, N_ext)] += tensor[axslice(axis, 0, pad_left)]
    #return center


class PeriodicConv3D(tf.keras.Model):
    """3D convolution layer with periodic padding."""

    def __init__(self, filters, kernel_size, kernel_center, strides=(1,1,1), **kw):
        super().__init__()
        # Store inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.strides = strides
        # Calculate pads
        self.pad_left = kernel_center
        self.pad_right = [ks - kc - 1 for ks, kc in zip(kernel_size, kernel_center)]
        # Build valid convolution
        self.conv_valid = tf.keras.layers.Conv3D(filters, kernel_size, strides=strides, padding='valid', **kw)

    def check_input_shape(self, input_shape):
        # Check strides evenly divide data shape
        batch, *data_shape, channels = input_shape
        for n, s in zip(data_shape, self.strides):
            if n%s != 0:
                raise ValueError("Strides must evenly divide data shape in periodic convolution.")

    def __call__(self, x):
        # Check shape
        self.check_input_shape(x.shape)
        # Iteratively apply periodic padding, skipping first axis (batch)
        for axis in range(3):
            x = pad_axis_periodic(x, axis+1, self.pad_left[axis], self.pad_right[axis])
        # Apply valid convolution
        return self.conv_valid(x)


class PeriodicConv3DTranspose(tf.keras.Model):
    """3D transposed convolution layer with periodic padding."""

    def __init__(self, filters, kernel_size, kernel_center, strides=(1, 1, 1), **kw):
        super().__init__()
        # Store inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_center = kernel_center
        self.strides = strides
        # Calculate pads
        self.pad_left = kernel_center
        self.pad_right = [ks - kc - 1 for ks, kc in zip(kernel_size, kernel_center)]
        self.output_padding = [[0, 0]] + [[0, min(ks, s) - 1] for ks, s in zip(kernel_size, strides)] + [[0, 0]]
        self.conv_valid = tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding='valid', **kw)

    def __call__(self, x):
        # Apply valid convolution
        x = self.conv_valid(x)
        # Pad with zeros to original expanded size
        x = tf.pad(x, self.output_padding)
        # Additively compress periodic padding, skipping first axis (batch)
        for axis in range(3):
            x = compress_axis_periodic(x, axis+1, self.pad_left[axis], self.pad_right[axis])
        return x


class Unet(tf.keras.Model):

    def __init__(self):
        super(Unet, self).__init__()

        filters = 16
        kernel_size = (3, 3, 3)
        kernel_center = (1, 1, 1)
        activation = 'relu'
        strides = (2, 2, 2)
        output_channels = 6

        def stack_down(stacksize):
            """Build convolution stack with downsampling on the last."""
            stack = []
            for i in range(stacksize - 1):
                stack.append(PeriodicConv3D(filters, kernel_size, kernel_center, activation=activation))
            stack.append(PeriodicConv3D(filters, kernel_size, kernel_center, activation=activation, strides=strides))
            return stack

        def stack_up(stacksize):
            """Build convolution stack with upsampling on the first."""
            stack = []
            stack.append(PeriodicConv3DTranspose(filters, kernel_size, kernel_center, activation=activation, strides=strides))
            for i in range(stacksize - 1):
                stack.append(PeriodicConv3D(filters, kernel_size, kernel_center, activation=activation))
            return stack

        self.down_stacks = [stack_down(3), stack_down(3)]
        self.up_stacks = [stack_up(3), stack_up(3)]
        self.Loutputs = PeriodicConv3D(output_channels, (1, 1, 1), (0, 0, 0), activation=activation)

        # self.Lc21 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc22 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lp23 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', strides=(2, 2), padding='same')

        # self.Lc31 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc32 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lp33 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', strides=(2, 2), padding='same')

        # self.Lc41 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc42 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lp43 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', strides=(2, 2), padding='same')

        # self.Lc51 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc52 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lp53 = tf.keras.layers.Conv2D(16, (6, 6), activation='linear', strides=(4, 4), padding='same')

        # self.LcZ1 = tf.keras.layers.Conv2D(16, (16, 16), activation='linear', padding='same')
        # self.LcZ2 = tf.keras.layers.Conv2D(16, (16, 16), activation='linear', padding='same')

        # self.Lu61 = tf.keras.layers.Conv2DTranspose(16, (6, 6), activation='linear', strides=(4, 4), padding='same')
        # self.Lc62 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc63 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')

        # self.Lu71 = tf.keras.layers.Conv2DTranspose(16, (2, 2), activation='linear', strides=(2, 2), padding='same')
        # self.Lc72 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc73 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')

        # self.Lu81 = tf.keras.layers.Conv2DTranspose(16, (2, 2), activation='linear', strides=(2, 2), padding='same')
        # self.Lc82 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc83 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')

        # self.Lu91 = tf.keras.layers.Conv2DTranspose(16, (2, 2), activation='linear', strides=(2, 2), padding='same')
        # self.Lc92 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc93 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')

        # self.Lu101 = tf.keras.layers.Conv2DTranspose(16, (2, 2), activation='linear', strides=(2, 2), padding='same')
        # self.Lc102 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')
        # self.Lc103 = tf.keras.layers.Conv2D(16, (3, 3), activation='linear', padding='same')

    def call(self, x):

        def eval_down(x):
            """Evaluate down unet stacks, saving partials before each downsampling."""
            partials = []
            for stack in self.down_stacks:
                for layer in stack[:-1]:
                    x = layer(x)
                partials.append(x)
                x = stack[-1](x)
            return partials, x

        def eval_up(x, partials):
            """Evaluate up unet stacks, concatenating partials after each upsampling."""
            for stack in self.up_stacks:
                x = stack[0](x)
                tf.concat([x, partials.pop()], axis=4)
                for layer in stack[1:]:
                    x = layer(x)
            return x

        pd, x = eval_down(x)
        x = eval_up(x, pd)
        return self.Loutputs(x)

    def cost_function(self, in_Net, labels):
        alpha = self.call(in_Net)
        return tf.reduce_mean((alpha - labels)**2), alpha

