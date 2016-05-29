import six.moves.cPickle as pickle
import numpy as np


def get_cifar10():
    file_train = []
    for i in [1, 2, 3, 4]:
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
    return train_x, train_y, val_x, val_y, test_x, test_y
