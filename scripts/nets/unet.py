"""Define u-net."""

"""
Issues:
    - Why did the upconvolutions have a different kernel size?
"""

import tensorflow as tf

# def pad_circular_nd(x: torch.Tensor, pad: int, dim) -> torch.Tensor:
#     """
#     :param x: shape [H, W]
#     :param pad: int >= 0
#     :param dim: the dimension over which the tensors are padded
#     :return:
#     """

#     if isinstance(dim, int):
#         dim = [dim]

#     for d in dim:
#         if d >= len(x.shape):
#             raise IndexError(f"dim {d} out of range")

#         idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
#         x = torch.cat([x, x[idx]], dim=d)

#         idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
#         x = torch.cat([x[idx], x], dim=d)
#         pass

#     return


class Unet(tf.keras.Model):

    def __init__(self):
        super(Unet, self).__init__()

        filters = 16
        kernel_size = (3, 3, 3)
        activation = 'relu'
        padding = 'same'
        strides = (2, 2, 2)
        output_channels = 6

        self.Lc11 = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, padding=padding)
        self.Lc12 = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, padding=padding)
        self.Lp13 = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, padding=padding, strides=strides)

        self.Lu101 = tf.keras.layers.Conv3DTranspose(filters, kernel_size, activation=activation, padding=padding, strides=strides)
        self.Lc102 = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, padding=padding)
        self.Lc103 = tf.keras.layers.Conv3D(filters, kernel_size, activation=activation, padding=padding)

        self.Loutputs = tf.keras.layers.Conv3D(output_channels, (1, 1, 1), activation=activation)

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



    def call(self, in_Net):

        c1 = self.Lc11(in_Net)
        c1 = self.Lc12(c1)
        p1 = self.Lp13(c1)

        c9 = p1
        u10 = self.Lu101(c9)
        u10 = tf.concat([u10, c1], axis=4)
        c10 = self.Lc102(u10)
        c10 = self.Lc103(c10)

        outputs = self.Loutputs(c10)
        return outputs

        # c2 = self.Lc21(p1)
        # c2 = self.Lc22(c2)
        # p2 = self.Lp23(c2)

        # c3 = self.Lc31(p2)
        # c3 = self.Lc32(c3)
        # p3 = self.Lp33(c3)

        # c4 = self.Lc41(p3)
        # c4 = self.Lc42(c4)
        # p4 = self.Lp43(c4)

        # c5 = self.Lc51(p4)
        # c5 = self.Lc52(c5)
        # p5 = self.Lp53(c5)

        # z1 = self.LcZ1(p5)
        # z1 = self.LcZ2(z1)

        # u6 = self.Lu61(z1)
        # u6 = tf.concat([u6, c5], axis=3)
        # c6 = self.Lc62(u6)
        # c6 = self.Lc63(c6)

        # u7 = self.Lu71(c6)
        # u7 = tf.concat([u7, c4], axis=3)
        # c7 = self.Lc72(u7)
        # c7 = self.Lc73(c7)

        # u8 = self.Lu81(c7)
        # u8 = tf.concat([u8, c3], axis=3)
        # c8 = self.Lc82(u8)
        # c8 = self.Lc83(c8)

        # u9 = self.Lu91(c8)
        # u9 = tf.concat([u9, c2], axis=3)
        # c9 = self.Lc92(u9)
        # c9 = self.Lc93(c9)


    def cost_function(self, in_Net, labels):
        alpha = self.call(in_Net)
        return tf.reduce_mean((alpha - labels)**2), alpha

