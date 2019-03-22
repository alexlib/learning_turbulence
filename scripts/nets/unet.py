"""Define u-net."""

"""
Issues:
    - Why did the upconvolutions have a different kernel size?
    - Why all linear activation?
"""

import numpy as np
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

    def __init__(self, stacks, stack_width, filters, output_channels, kernel_size=(3,3,3), kernel_center=None, strides=(2,2,2), **kw):
        """
        Parameters
        ----------
        stacks : int
            Number of unet stacks.
        stackwidth : int
            Number of convolutions per stack.
        filters : int
            Number of intermediate convolution filters.
        output_channels : int
            Number of output channels
        kernel_size : int or tuple of ints
            Kernel size for each dimension, default: (3,3,3)
        kernel_center : int or tuple of ints
            Kernel center for each dimension, default: kernel_size//2
        strides : tuple of ints
            Strides for each dimension, default: (2,2,2)

        Other keyword arguments are passed to the convolution layers.

        """
        super().__init__()

        # Handle integer kernel specifications
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(kernel_center, int):
            kernel_center = (kernel_center,) * 3
        if kernel_center is None:
            kernel_center = tuple(ks//2 for ks in kernel_size)
        print(kernel_center)

        def stack_down(stacksize):
            """Build convolution stack with downsampling on the last."""
            stack = []
            for i in range(stacksize - 1):
                stack.append(PeriodicConv3D(filters, kernel_size, kernel_center, **kw))
            stack.append(PeriodicConv3D(filters, kernel_size, kernel_center, strides=strides, **kw))
            return stack

        def stack_up(stacksize):
            """Build convolution stack with upsampling on the first."""
            stack = []
            stack.append(PeriodicConv3DTranspose(filters, kernel_size, kernel_center, strides=strides, **kw))
            for i in range(stacksize - 1):
                stack.append(PeriodicConv3D(filters, kernel_size, kernel_center, **kw))
            return stack

        self.down_stacks = [stack_down(stack_width) for i in range(stacks)]
        self.up_stacks = [stack_up(stack_width) for i in range(stacks)]
        self.outlayer = PeriodicConv3D(output_channels, (1, 1, 1), (0, 0, 0), **kw)

    def call(self, x_list):
        """Call unet on multiple inputs, combining at bottom."""

        def eval_down(x):
            """Evaluate down unet stacks, saving partials before each downsampling."""
            partials = []
            for stack in self.down_stacks:
                for layer in stack[:-1]:
                    x = layer(x)
                partials.append(x)
                x = stack[-1](x)
            return x, partials

        def eval_up(x, partials):
            """Evaluate up unet stacks, concatenating partials after each upsampling."""
            for stack in self.up_stacks:
                x = stack[0](x)
                x = tf.concat([x, partials.pop()], axis=4)
                for layer in stack[1:]:
                    x = layer(x)
            return x

        # Evaluate down for each input
        x, partials = zip(*[eval_down(x) for x in x_list])
        # Concatenate results and partials for each input
        x = tf.concat(x, axis=4)
        partials = list(zip(*partials))
        partials = [tf.concat(p, axis=4) for p in partials]
        # Evaluate up
        x = eval_up(x, partials)
        return self.outlayer(x)

    def cost_function(self, inputs, labels):
        outputs = self.call(inputs)
        return tf.reduce_mean((outputs - labels)**2), outputs

