'''mnist.py
Loads and preprocesses the MNIST dataset
Xavier Markowitz and Austin Perera
CS443: Bio-Inspired Machine Learning
Project 1: Hebbian Learning
'''
import os
import numpy as np
from PIL import Image

def get_mnist(N_val, scale_fact, path='data/mnist'):
    '''Load and preprocesses the MNIST dataset (train and test sets) located on disk within `path`.

    Parameters:
    -----------
    N_val: int. Number of data samples to reserve from the training set to form the validation set. As usual, each
    sample should be in EITHER training or validation sets, NOT BOTH.
    path: str. Path in working directory where MNIST dataset files are located.

    Returns:
    -----------
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels)
    '''
    x_test = np.load(path+"/x_test.npy")
    y_test = np.load(path+"/y_test.npy")
    x_train_temp = np.load(path+"/x_train.npy")
    y_train_temp = np.load(path+"/y_train.npy")
    x_t, y_train, x_val, y_val = train_val_split(x_train_temp, y_train_temp, N_val)
    x_test = preprocess_mnist(x_test, scale_fact)
    x_train = preprocess_mnist(x_t, scale_fact)
    x_val = preprocess_mnist(x_val, scale_fact)
    return x_train, y_train, x_test, y_test, x_val, y_val


def preprocess_mnist(x, scale_fact):
    '''Preprocess the data `x` so that:
    - the maximum possible value in the dataset is 1 (and minimum possible is 0).
    - the shape is in the format: `(N, M)`

    Parameters:
    -----------
    x: ndarray. shape=(N, I_y, I_x). MNIST data samples represented as grayscale images.

    Returns:
    -----------
    ndarray. shape=(N, I_y*I_x). MNIST data samples represented as MLP-compatible feature vectors.
    '''
    x = resize_images(x, scale_fact)
    x2 = (x - np.min(x))/(np.max(x)-np.min(x))

    return(np.reshape(x2, (x2.shape[0],(x2.shape[1]*x2.shape[2]))))

def resize_images(imgs, scale_fact=3):
    ''' Rescales collection of images represented as a single ndarray

    Parameters:
    -----------
    imgs: ndarray. shape = (num images, x, y, color chan)
    scale_factor: downscale image resolution by this amount

    Returns:
    -----------
    scaled_imgs: ndarray. the downscaled images.
    '''
    if scale_fact == 1.0:
        print(f'preprocess_images: No resizing to do, scale factor = {scale_fact}.')
        return imgs

    # print(f'Resizing {len(imgs)} images to {16//scale_fact}x{16//scale_fact}...', end='')

    num_imgs = imgs.shape[0]
    scaled_imgs = np.zeros([num_imgs, 28//scale_fact, 28//scale_fact], dtype=np.uint8)

    for i in range(num_imgs):
        currImg = Image.fromarray(imgs[i, :])
        currImg = currImg.resize(size=(28//scale_fact, 28//scale_fact))
        scaled_imgs[i, :] = np.array(currImg, dtype=np.uint8)

    print('Done!')
    return scaled_imgs

def train_val_split(x, y, N_val):
    '''Divide samples into train and validation sets. As usual, each sample should be in EITHER training or validation
    sets, NOT BOTH. Data samples are already shuffled.

    Parameters:
    -----------
    x: ndarray. shape=(N, M). MNIST data samples represented as vector vectors
    y: ndarray. ints. shape=(N,). MNIST class labels.
    N_val: int. Number of data samples to reserve from the training set to form the validation set.

    Returns:
    -----------
    x: ndarray. shape=(N-N_val, M). Training set.
    y: ndarray. shape=(N-N_val,). Training set class labels.
    x_val: ndarray. shape=(N_val, M). Validation set.
    y_val ndarray. shape=(N_val,). Validation set class labels.
    '''
    x_t = x[N_val:,:]
    y_t = y[N_val:]
    x_val = x[:N_val,:]
    y_val = y[:N_val]
    return x_t, y_t, x_val, y_val
