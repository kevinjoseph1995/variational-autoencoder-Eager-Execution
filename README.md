# Variational Autoencoder  
  - A variational autoencoder model trained on the MNIST dataset using Tensorflow's Eager Execution, an imperative programming environment that evaluates operations immediately.
  - Eager Execution is different from the computational graph approach used within Tensorflow.  
  - https://www.tensorflow.org/programmers_guide/eager

## Model details
 - The encoder maps the input image to a latent normal distribution characterized by its mean and diagonal covariance matrix.
 - The decoder samples from this distribution using the reparametrization trick and maps the latent vector to the corresponding reconstruction.
 - For more details about the implemented model inspect the encoder() and decoder() class in var_autoencoder.py. 
 - Refer https://arxiv.org/pdf/1606.05908.pdf for an extensive review about variational autoencoders. 
## Training the model
```sh
$ python var_autoencoder.py
```
After completion of training, the model is stored within var_checkpoint. 

## Generating images.
```sh
$ python var_generator.py
```
The generated images are stored in var_autoencoder_reconstructed_imgs.  
## ##
![alt text](https://github.com/kevinjoseph1995/variational-autoencoder-Eager-Execution/blob/master/generated_images.png)

Some of the generated images after the model was trained for 20 epochs.
