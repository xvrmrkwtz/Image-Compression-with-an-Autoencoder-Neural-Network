'''preprocess_data.py
Preprocessing data in STL-10 image dataset
YOUR NAMES HERE
CS343: Neural Networks
Project 2: Multilayer Perceptrons
'''
import numpy as np
import load_stl10_dataset


def standardize_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    TODO:
    1) Cast imgs to float64
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Treating the pixels as features, standardize the features "seperately"
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    imgs = imgs.astype('float64')
    #imgs = np.prod(imgs, axis = (1,2,3), keepdims = True)
    # imgs = np.reshape(imgs, (imgs.shape[0], np.prod(imgs.shape[1:])))
    imgs = imgs.transpose((0,3,1,2))
    #print(imgs.shape)
    mean = imgs.mean(axis = 0)
    std = np.std(imgs, axis = 0)
    imgs = (imgs - mean)/std
    labels = labels - 1
    return imgs, labels, mean, std

def preprocess_stl(imgs, labels):
    '''Preprocesses stl image data for training by a MLP neural network

    Parameters:
    ----------
    imgs: unint8 ndarray  [0, 255]. shape=(Num imgs, height, width, RGB color chans)

    Returns:
    ----------
    imgs: float64 ndarray [0, 1]. shape=(Num imgs N,)
    Labels: int ndarray. shape=(Num imgs N,). Contains int-coded class values 0,1,...,9

    TODO:
    1) Cast imgs to float64
    2) Flatten height, width, color chan dims. New shape will be (num imgs, height*width*chans)
    3) Treating the pixels as features, standardize the features "seperately"
    4) Fix class labeling. Should span 0, 1, ..., 9 NOT 1,2,...10
    '''
    imgs = imgs.astype('float64')
    #imgs = np.prod(imgs, axis = (1,2,3), keepdims = True)
    # imgs = np.reshape(imgs, (imgs.shape[0], np.prod(imgs.shape[1:])))
    imgs = imgs.transpose((0,3,1,2))
    #print(imgs.shape)

    # imgs = (imgs/255) - .5
    labels = labels - 1
    return imgs, labels

# def get_standardization(imgs):
#     imgs = imgs.astype('float64')
#     imgs = imgs.transpose((0,3,1,2))
#     mean = imgs.mean(axis = 0)
#     std = np.std(imgs, axis = 0)
#     # mean = mean.transpose((1,0,2))
#     # std = std.transpose((1,0,2))
#     return mean, std

def create_splits(data, y, n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500):
    '''Divides the dataset up into train/test/validation/development "splits" (disjoint partitions)

    Parameters:
    ----------
    data: float64 ndarray. Image data. shape=(Num imgs, height*width*chans)
    y: ndarray. int-coded labels.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)

    TODO:
    1) Divvy up the images into train/test/validation/development non-overlapping subsets (see return vars)

    NOTE: Resist the urge to shuffle the data here! It is best to re-shuffle the data after
    each epoch of training so hold off on shuffling until you train your neural network.
    '''

    if n_train_samps + n_test_samps + n_valid_samps + n_dev_samps != len(data):
        samps = n_train_samps + n_test_samps + n_valid_samps + n_dev_samps
        print(f'Error! Num samples {samps} does not equal num images {len(data)}!')
        return

    x_train = data[:n_train_samps, :]
    y_train = y[:n_train_samps]
    x_test = data[n_train_samps: (n_train_samps + n_test_samps), :]
    y_test = y[n_train_samps: (n_train_samps + n_test_samps)]
    x_val = data[(n_train_samps + n_test_samps): (n_train_samps + n_test_samps + n_valid_samps), :]
    y_val = y[(n_train_samps + n_test_samps): (n_train_samps + n_test_samps + n_valid_samps)]
    x_dev = data[(n_train_samps + n_test_samps + n_valid_samps): (n_train_samps + n_test_samps + n_valid_samps+ n_dev_samps), :]
    y_dev = y[(n_train_samps + n_test_samps + n_valid_samps): (n_train_samps + n_test_samps + n_valid_samps+ n_dev_samps)]
    
    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev

def load_stl10_std(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500, scale_fact=3):
    '''Automates the process of:
    - loading in the STL-10 dataset and labels
    - preprocessing
    - creating the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''
    
    data, y = load_stl10_dataset.load(scale_fact = scale_fact)
    # mean, std = get_standardization(data)
    data, y, mean, std = preprocess_stl(data, y)
    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = create_splits(data, y, n_train_samps, n_test_samps, n_valid_samps, n_dev_samps)

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev, mean, std

def load_stl10_div(n_train_samps=3500, n_test_samps=500, n_valid_samps=500, n_dev_samps=500, scale_fact=3):
    '''Automates the process of:
    - loading in the STL-10 dataset and labels
    - preprocessing
    - creating the train/test/validation/dev splits.

    Returns:
    ----------
    None if error
    x_train (training samples),
    y_train (training labels),
    x_test (test samples),
    y_test (test labels),
    x_val (validation samples),
    y_val (validation labels),
    x_dev (development samples),
    y_dev (development labels)
    '''
    
    data, y = load_stl10_dataset.load(scale_fact = scale_fact)
    # mean, std = get_standardization(data)
    data, y= preprocess_stl(data, y)
    x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev = create_splits(data, y, n_train_samps, n_test_samps, n_valid_samps, n_dev_samps)

    return x_train, y_train, x_test, y_test, x_val, y_val, x_dev, y_dev