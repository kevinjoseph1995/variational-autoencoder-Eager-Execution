# Variational Autoencoder 
![alt text](https://github.com/kevinjoseph1995/variational-autoencoder-Eager-Execution/blob/master/generated_images.png)
  - A variational autoencoder model trained on the MNIST dataset using Tensorflow's Eager Execution, an imperative programming environment that evaluates operations immediately.
  - Eager Execution is different from the computational graph approach used within Tensorflow.  
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
