import tensorflow as tf
import dataset
import numpy as np
import os
import tensorflow.contrib.eager as tfe
import scipy.misc
tf.enable_eager_execution()

class autoencoder(tf.keras.Model):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.conv2=tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.conv3=tf.keras.layers.Conv2D(filters=100,kernel_size=7,strides=1,padding='valid',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.conv4=tf.keras.layers.Conv2D(filters=50,kernel_size=1,strides=1,padding='valid',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.upconv1=tf.keras.layers.Conv2D(filters=100,kernel_size=1,strides=1,padding='valid',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.upconv2=tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=7,strides=1,padding='valid',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.upconv3=tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.upconv4=tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
    def call(self, inputs, training=False):
        #input shape (samples, rows, cols, channels)
        l1=self.conv1(inputs)
        #print (l1.shape)
        l2=self.conv2(l1)
        #print (l2.shape)
        l3=self.conv3(l2)
        #print (l3.shape)
        l4=self.conv4(l3)
        #print (l4.shape)
        l5=self.upconv1(l4)
        #print (l5.shape)
        l6=self.upconv2(l5)
        #print (l6.shape)
        l7=self.upconv3(l6)
        #print (l7.shape)
        output=self.upconv4(l7)
        #print (output.shape)
        #output shape (samples, new_rows, new_cols, filters)
        return output
model=autoencoder()

def loss(model,input):
    output=model(input)
    input=tf.squeeze(input)
    output=tf.squeeze(output)
    return tf.reduce_mean(tf.reduce_sum(tf.square(input-output),axis=[1,2]))

def grad(model, input  ):
  with tf.GradientTape() as tape:
    loss_value = loss(model, input)
  return tape.gradient(loss_value, model.variables)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

checkpoint_dir = 'checkpoint/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

number_of_test_images=10
test_data=dataset.train('./datasets')
test_data=test_data.batch(number_of_test_images)
iterator = test_data.make_one_shot_iterator()
next_element,_ = iterator.get_next()
input=tf.reshape(next_element,[next_element.shape[0],28,28,1])#(samples, rows, cols, channels)
print("Final loss: {:.3f}".format(loss(model, input)))
output=model(input)
input_images=input.numpy()
output_images=output.numpy()
for i in range(input_images.shape[0]):
    input_image=np.squeeze(input_images[i,:,:,:])
    output_image=np.squeeze(output_images[i,:,:,:])
    print_image=np.hstack((input_image,output_image))
    file_name='autoencoder_reconstructed_imgs/'+str(i)+'.png'
    scipy.misc.imsave(file_name, print_image)
