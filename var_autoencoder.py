import tensorflow as tf
import dataset
import numpy as np
import os
import tensorflow.contrib.eager as tfe
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

training_data=dataset.train('./datasets')
training_data=training_data.shuffle(60000).repeat(10).batch(64)
iterator = training_data.make_one_shot_iterator()
next_element,_ = iterator.get_next()
input=tf.reshape(next_element,[next_element.shape[0],28,28,1])#(samples, rows, cols, channels)

def loss(var_encoder,var_decoder,input):
    mu,sigma=var_encoder(input)
    mu_zero=tf.zeros_like(mu)
    normal_distribution=tf.contrib.distributions.MultivariateNormalDiag(loc=mu_zero)
    Z=tf.multiply(normal_distribution.sample(),sigma)+mu
    Z=tf.reshape(Z,[Z.shape[0],1,1,Z.shape[1]])
    reconstruction=var_decoder(Z)

    K=mu.shape[1]
    K=K.value
    KL_divergence=tf.reduce_sum(sigma,axis=1)+tf.reduce_sum(tf.square(mu),axis=1)-K-tf.reduce_sum(tf.log(sigma+0.00000001),axis=1)
    KL_divergence=0.5*tf.reduce_mean(KL_divergence)

    Expectation=tf.reduce_mean(tf.reduce_sum(tf.square(tf.squeeze(input)-tf.squeeze(reconstruction)),axis=[1,2]))

    loss=KL_divergence+Expectation
    return loss

def grad(var_encoder,var_decoder,input):
  with tf.GradientTape() as tape:
    loss_value = loss(var_encoder,var_decoder,input)
  return tape.gradient(loss_value, [var_encoder.variables,var_decoder.variables])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

checkpoint_dir = 'var_checkpoint/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      var_encoder=var_encoder,
                      var_decoder=var_decoder,
                      optimizer_step=tf.train.get_or_create_global_step())
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

print("Initial loss: {:.3f}".format(loss(var_encoder,var_decoder,input)))

for (i, (next_element, _ )) in enumerate(training_data):
    input=tf.reshape(next_element,[next_element.shape[0],28,28,1])#(samples, rows, cols, channels)
    # Calculate derivatives of the input function with respect to its parameters.
    grads = grad(var_encoder,var_decoder,input)
    # Apply the gradient to the model
    optimizer.apply_gradients(zip(grads[0], var_encoder.variables),global_step=tf.train.get_or_create_global_step())
    optimizer.apply_gradients(zip(grads[1], var_decoder.variables),global_step=tf.train.get_or_create_global_step())
    if i % 200 == 0:
        print("Loss at step {:04d}: {:.3f}".format(i, loss(var_encoder,var_decoder,input)))


print("Final loss: {:.3f}".format(loss(var_encoder,var_decoder,input)))
root.save(file_prefix=checkpoint_prefix)
