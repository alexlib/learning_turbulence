

import tensorflow as tf
import numpy as np



tf.enable_eager_execution()

execfile("Unet.py")



RESTORE = False
train_batch_size = 1
num_features = 128


Unet_obj = Unet()







optimizer = tf.train.AdamOptimizer(1.0e-3)
checkpoint_path = "checkpoints/model_Unet"

if (RESTORE==True):
	Unet_obj.load_weights(checkpoint_path)


labels = np.zeros((train_batch_size,num_features, num_features,1))

Net_input = np.zeros((train_batch_size,num_features, num_features,1))


with tf.device('/gpu:0'):
    for epoch in range(1):
        for train_iter in range(1):

            #loads labels and network inputs here and asign to labels and Net_input

            tf_labels = tf.cast( labels, datatype)


            tf_Net_input = tf.cast( Net_input , datatype)



            with tf.GradientTape() as tape:
                tape.watch(Unet_obj.variables)
                cost_value , alpha_net = Unet_obj.cost_function(tf_Net_input , tf_labels )
            weight_grads = tape.gradient(cost_value, [Unet_obj.variables] )

            #clipped_grads = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
            optimizer.apply_gradients(zip(weight_grads, Unet_obj.variables), global_step=tf.train.get_or_create_global_step())

            print( epoch, train_iter , cost_value.numpy() )

            if (((train_iter+1)%10)==0):
                print( "saving weights." )
                RT.save_weights("checkpoints/model_Unet")





