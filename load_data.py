from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import time
import six.moves.cPickle as pickle


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def seperate_data(data_x, data_y, y_start_from_zero=True):
    assert len(data_x) == len(data_y)
    index1 = []
    index2 = []
    for i in range(len(data_y)):
        if data_y[i] <= 4:
            index1.append(i)
        else:
            index2.append(i)
    index1 = np.array(index1)
    index2 = np.array(index2)
    data_1_x = data_x[index1]
    data_1_y = data_y[index1]

    data_2_x = data_x[index2]
    data_2_y = data_y[index2]
    if y_start_from_zero:
        data_1_y -= 5
        data_2_y -= 5
        
    return data_1_x, data_1_y, data_2_x, data_2_y


def load_dataset_seperate(get_first_part=True):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    X_train_1, y_train_1, X_train_2, y_train_2 = seperate_data(X_train, y_train)
    X_val_1, y_val_1, X_val_2, y_val_2 = seperate_data(X_val, y_val)
    X_test_1, y_test_1, X_test_2, y_test_2 = seperate_data(X_test, y_test)
    if get_first_part:
        return X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1
    else:
        return X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2


def get_cifar10():
    file_train = []
    for i in [1]:
        file_train.append('cifar-10-batches-py/data_batch_' + str(i))
    file_val = 'cifar-10-batches-py/data_batch_5'
    file_test = 'cifar-10-batches-py/test_batch'
    train_x = np.zeros((0, 3, 32, 32))
    train_y = []

    shape = (-1, 3, 32, 32)
    for fi in file_train:
        with open(fi, 'rb') as f:
            var = pickle.load(f, encoding='latin1')
            train_x = np.append(train_x, var['data'].reshape(shape), axis=0)
            train_y.extend(var['labels'])
    train_y = np.array(train_y, dtype=np.uint8)

    with open(file_val, 'rb') as f:
        var = pickle.load(f, encoding='latin1')
        val_x = var['data'].reshape(shape)
        val_y = np.array(var['labels'], dtype=np.uint8)

    with open(file_test, 'rb') as f:
        var = pickle.load(f, encoding='latin1')
        test_x = var['data'].reshape(shape)
        test_y = np.array(var['labels'], dtype=np.uint8)

    train_x = train_x / np.float32(256)
    val_x = val_x / np.float32(256)
    test_x = test_x / np.float32(256)

    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    test_x = test_x.astype(np.float32)

    all_x = np.concatenate((train_x, val_x, test_x), axis=0)
    color_mean = all_x.mean(axis=(0, 2, 3))
    train_x[:, 0, :, :] -= color_mean[0]
    train_x[:, 1, :, :] -= color_mean[1]
    train_x[:, 2, :, :] -= color_mean[2]
    val_x[:, 0, :, :] -= color_mean[0]
    val_x[:, 1, :, :] -= color_mean[1]
    val_x[:, 2, :, :] -= color_mean[2]
    test_x[:, 0, :, :] -= color_mean[0]
    test_x[:, 1, :, :] -= color_mean[1]
    test_x[:, 2, :, :] -= color_mean[2]

    return train_x, train_y, val_x, val_y, test_x, test_y
