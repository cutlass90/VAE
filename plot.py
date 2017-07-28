import random
import os

import numpy as np 
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from vae import VAE

input_dim = 28*28
z_dim = 2
load_path = 'models/'

model = VAE(input_dim=784, z_dim=2, do_train=False)
model.load_model(load_path, model.sess)

def to_image(x):
    return x.reshape(28, 28)

def reconstruct(k=4):
    original = random.sample(list(mnist.validation.images), k)
    reconstruction = list(model.reconstruct(original))

    fig = plt.figure(figsize=(4, k*2))
    for orig, rec, i in zip(original, reconstruction, range(k)):
        ax = fig.add_subplot(k, 2, 2*i+1)
        ax.imshow(to_image(orig), cmap='gray')
        ax.set_title('original')
        ax.set_axis_off()
        ax = fig.add_subplot(k, 2, 2*i+2)
        ax.imshow(to_image(rec), cmap='gray')
        ax.set_title('reconstructed')
        ax.set_axis_off()
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/reconstruction.png')
    plt.close()
    print('Reconstruction saved.')

def sample():

    fig = plt.figure(figsize=(7, 7))
    for i in range(225):
        ax = fig.add_subplot(15, 15, i+1)
        z = np.random.normal(size=[1, z_dim]).astype(np.float32)
        x = model.reconstruct_from_z(z)
        ax.imshow(to_image(x), cmap='gray', aspect='auto')
        ax.set_axis_off()

    #remove spacings between subplots
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/sample_from_latent_space.png')
    plt.close()
    print('Sample saved.')

def plot_latent_space():
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny)

    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z_mu = [[xi, yi]]
            x_mean = model.reconstruct_from_z(z_mu)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = to_image(x_mean[0])

    fig = plt.figure(figsize=(5, 5))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    ax = fig.add_subplot(111)
    ax.imshow(canvas,  origin='upper', cmap='gray', interpolation='bicubic')
    ax.set_axis_off()
    os.makedirs('pics', exist_ok=True)
    fig.savefig('pics/latent_space.png')
    plt.close()
    print('Latent space saved.')

    
if __name__ == '__main__':
    sample()
    plot_latent_space()
