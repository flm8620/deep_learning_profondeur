from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from load_data import *
from lenet5 import *
# from cifar10_nin import *
import six.moves.cPickle as pickle


def main(model, model_file, layer_name):
    batch_size = 64
    seperate = True
    load_first_part = False
    print('batch_size=', batch_size)
    # Load the dataset
    print("Loading data...")
    if model == 'cifar':
        X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10()
    elif model == 'lenet':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    else:
        assert False

    if seperate:
        X_train_1, y_train_1, X_train_2, y_train_2 = seperate_data(X_train, y_train)
        X_val_1, y_val_1, X_val_2, y_val_2 = seperate_data(X_val, y_val)
        X_test_1, y_test_1, X_test_2, y_test_2 = seperate_data(X_test, y_test)
        if load_first_part:
            X_train, y_train, X_val, y_val, X_test, y_test = X_train_1, y_train_1, X_val_1, y_val_1, X_test_1, y_test_1
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2

    print(len(X_train), 'train images')
    print(len(X_val), 'val images')
    print(len(X_test), 'test images')

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    if model == 'lenet':
        net = build_lenet5(input_var)
        network = net['output']
    elif model == 'cifar':
        pass
        # net = cifar10_nin.build_model2(input_var)
        # network = net['output']
    else:
        print("Unrecognized model type %r." % model)
        return
    if model_file is not None:
        with np.load(model_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
    else:
        assert False

    # middle_output = theano.function([input_var], net[layer_name].output)
    print("Getting middle output...")

    output = lasagne.layers.get_output(net[layer_name])
    get_output = theano.function([input_var], output.flatten(2))

    output_shape = np.array(lasagne.layers.get_output_shape(net[layer_name]))
    print('layer '+layer_name+' shape :', output_shape)
    output_size = np.prod(output_shape[1:])
    all_train_output = []
    all_train_y = []
    all_test_output = []
    all_test_y = []
    print('getting from train')
    for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
        print('.',end='',flush=True)
        inputs, targets = batch
        batch_output = get_output(inputs)  # a numpy ndarray
        all_train_output.extend(batch_output.tolist())
        all_train_y.extend(targets.tolist())
    print()
    print('getting from test')
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        print('.', end='', flush=True)
        inputs, targets = batch
        batch_output = get_output(inputs)  # a numpy ndarray
        all_test_output.extend(batch_output.tolist())
        all_test_y.extend(targets.tolist())
    print()

    print("train output shape : ", np.array(all_train_output).shape)
    print("train y shape : ", np.array(all_train_y).shape)
    print("test output shape : ", np.array(all_test_output).shape)
    print("test y shape : ", np.array(all_test_y).shape)

    filename = model + '_' + layer_name + '_output.save'
    with open(filename, 'wb') as f:
        pickle.dump([all_train_output, all_train_y, all_test_output, all_test_y], f)
    print('... saved to ', filename)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s MODEL MODEL_FILE LAYER_NAME" % sys.argv[0])
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['model_file'] = sys.argv[2]
        if len(sys.argv) > 3:
            kwargs['layer_name'] = sys.argv[3]
        main(**kwargs)
