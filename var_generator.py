import tensorflow as tf
import dataset
import numpy as np
import os
import tensorflow.contrib.eager as tfe
import scipy.misc
tf.enable_eager_execution()

class encoder(tf.keras.Model):
    def __init__(self):
        super(encoder, self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=16,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.conv2=tf.keras.layers.Conv2D(filters=32,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.conv3=tf.keras.layers.Conv2D(filters=100,kernel_size=7,strides=1,padding='valid',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
    def call(self, inputs, training=False):
        #input shape (samples, rows, cols, channels)
        l1=self.conv1(inputs)
        l2=self.conv2(l1)
        output=self.conv3(l2)
        output=tf.squeeze(output)
        mu=output[:,:50]
        sigma=output[:,50:]
        #mu shape (samples, latent_dimension_size)
        #sigma shape (samples, latent_dimension_size)
        return mu,sigma
class decoder(tf.keras.Model):
    def __init__(self):
        super(decoder, self).__init__()
        self.upconv1=tf.keras.layers.Conv2DTranspose(filters=32,kernel_size=7,strides=1,padding='valid',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.upconv2=tf.keras.layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
        self.upconv3=tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,strides=2,padding='same',data_format='channels_last',dilation_rate=1,activation='relu',use_bias=True)
    def call(self,input, training=False):
        #print('Decoder')
        #print (input.shape)
        l1=self.upconv1(input)
        #print (l1.shape)
        l2=self.upconv2(l1)
        #print (l2.shape)
        output=self.upconv3(l2)
        return output
var_encoder=encoder()
var_decoder=decoder()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
checkpoint_dir = 'var_checkpoint/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      var_encoder=var_encoder,
                      var_decoder=var_decoder,
                      optimizer_step=tf.train.get_or_create_global_step())
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

def generate(num_images):
    mu_zero=tf.zeros([num_images,50])
    normal_distribution=tf.contrib.distributions.MultivariateNormalDiag(loc=mu_zero)
    Z=normal_distribution.sample()
    Z=tf.reshape(Z,[Z.shape[0],1,1,Z.shape[1]])
    generated_imgs=var_decoder(Z)
    return generated_imgs
generated_images=generate(50)
generated_images=generated_images.numpy()
for i in range(generated_images.shape[0]):
    generated_image=np.squeeze(generated_images[i,:,:,:])
    file_name='var_autoencoder_reconstructed_imgs/'+str(i)+'.png'
    scipy.misc.imsave(file_name, generated_image)
