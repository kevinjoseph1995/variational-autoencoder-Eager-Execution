import tensorflow as tf
import dataset
import numpy as np
tf.enable_eager_execution()




class variational_autoencoder(tf.keras.Model):
    def __init__(self):
        super(variational_autoencoder, self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=4,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.conv2=tf.keras.layers.Conv2D(filters=8,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.flattened=tf.keras.layers.Flatten()
        self.reshape=tf.keras.layers.Reshape(target_shape=())

    def call(self, inputs, training=False):
        #input shape (samples, rows, cols, channels)
        print (inputs.shape)
        l1=self.conv1(inputs)
        print (l1.shape)
        l2=self.conv2(l1)
        print (l2.shape)
        output=self.flattened(l2)
        print (output.shape)

        #output shape (samples, new_rows, new_cols, filters)
        return output
model=variational_autoencoder()


training_data=dataset.train('./datasets')
training_data=training_data.shuffle(60000).repeat(4).batch(32)
iterator = training_data.make_one_shot_iterator()
next_element = iterator.get_next()
input=tf.reshape(next_element[0],[next_element[0].shape[0],28,28,1])

output=model(input)
