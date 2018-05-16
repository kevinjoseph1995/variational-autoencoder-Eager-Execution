import tensorflow as tf
import dataset
import numpy as np
tf.enable_eager_execution()

class variational_autoencoder(tf.keras.Model):
    def __init__(self):
        super(variational_autoencoder, self).__init__()
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
model=variational_autoencoder()

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


training_data=dataset.train('./datasets')
training_data=training_data.shuffle(60000).repeat(10).batch(32)

iterator = training_data.make_one_shot_iterator()

next_element,_ = iterator.get_next()
input=tf.reshape(next_element,[next_element.shape[0],28,28,1])#(samples, rows, cols, channels)
print(loss(model, input).shape)

print("Initial loss: {:.3f}".format(loss(model, input)))

for (i, (next_element, _ )) in enumerate(training_data):
    input=tf.reshape(next_element,[next_element.shape[0],28,28,1])#(samples, rows, cols, channels)
    # Calculate derivatives of the input function with respect to its parameters.
    grads = grad(model,input)
    # Apply the gradient to the model
    optimizer.apply_gradients(zip(grads, model.variables),global_step=tf.train.get_or_create_global_step())
    if i % 200 == 0:
        print("Loss at step {:04d}: {:.3f}".format(i, loss(model, input)))

print("Final loss: {:.3f}".format(loss(model, input)))
