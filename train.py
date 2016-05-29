#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import cifar10_nin
from lenet5 import *
import time
from load_data import *






# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='lenet', num_epochs=20, model_file=None):
    batch_size = 64
    seperate = True
    if seperate:
        nOutput=5
    else:
        nOutput=10
    load_first_part = True
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
        net = build_lenet5(input_var,nOutput)
        network = net['output']
    elif model == 'cifar':
        # pass
        net = cifar10_nin.build_model2(input_var,nOutput)
        network = net['output']
    else:
        print("Unrecognized model type %r." % model)
        return
    if model_file is not None:
        with np.load(model_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting training...")

    for epoch in range(num_epochs):
        time_epoch = time.time()

        train_err = 0
        train_batches = 0
        start_time = time.time()
        print("Training stage:")
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            time_batch = time.time()
            inputs, targets = batch
            this_train_err = train_fn(inputs, targets)
            train_err += this_train_err
            train_batches += 1
            print('train batch', train_batches, 'err+=', this_train_err, time.time() - time_batch, 'seconds')

        val_err = 0
        val_acc = 0
        val_batches = 0
        print("Validation stage:")
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            time_batch = time.time()
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            print('valid batch', val_batches, 'err+=', err, 'acc+=', acc, time.time() - time_batch, 'seconds')

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        # Optionally, you could now dump the network weights to a file like this:
        print('model saved to ' + model + '_model' + str(epoch) + '.npz')
        np.savez(model + '_model' + str(epoch) + '.npz', *(lasagne.layers.get_all_param_values(network)))
        print('epoch_time ', (time.time() - time_epoch) / 60., 'minutes')

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        time_batch = time.time()
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
        print('test batch', test_batches, 'err+=', err, 'acc+=', acc, time.time() - time_batch, 'seconds')
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['model_file'] = sys.argv[3]
        main(**kwargs)
