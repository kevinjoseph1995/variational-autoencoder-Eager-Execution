# Variational Autoencoder 
  - A variational autoencoder model trained on the MNIST dataset using Tensorflow's Eager Execution, an imperative programming environment that evaluates operations immediately.
  - https://www.tensorflow.org/programmers_guide/eager

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
