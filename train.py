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
import argparse

import numpy as np
import theano
import theano.tensor as T

import lasagne
import cifar10_nin
from lenet5 import *
from load_data import *


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name", choices=['cifar', 'lenet'])
    parser.add_argument('-n', '--num-epochs', type=int, default=20)
    parser.add_argument('-f', '--model-file', help="model file")
    parser.add_argument('--half', help='split the data and take first part', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.01)

    args = parser.parse_args()

    model = args.model
    batch_size = args.batch_size
    separate = args.half
    model_file = args.model_file
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    save_file_name = model + '_model'
    if separate:
        save_file_name = 'half_' + save_file_name
        nOutput = 5
    else:
        nOutput = 10
    load_first_part = True
    print('--Parameter--')
    print('  model=', model)
    print('  batch_size=', batch_size)
    print('  num_epochs=', num_epochs)
    print('  learning_rate=', learning_rate)
    print('  separate data : ', separate)
    print('  model_file : ', model_file)
    print('  nOutput = ', nOutput)
    print('  model will be saved to : ', save_file_name + '*.npz')
    print()

    print("Loading data...")
    if model == 'cifar':
        X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10()
    elif model == 'lenet':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    else:
        assert False

    if separate:
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
        net = build_lenet5(input_var, nOutput)
        network = net['output']
    elif model == 'cifar':
        # pass
        net = cifar10_nin.build_model2(input_var, nOutput)
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
        loss, params, learning_rate=learning_rate, momentum=0.9)

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
            print('train batch', train_batches, 'err+=', '{:.5f}'.format(this_train_err.item()),
                  '{:.2f}'.format(time.time() - time_batch), 'seconds')

        val_err = 0
        val_acc = 0
        val_batches = 0
        print("Validation stage ..")
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        # Optionally, you could now dump the network weights to a file like this:

        model_file = save_file_name + str(epoch) + '.npz'
        print('model saved to ' + model_file)
        np.savez(model_file, *(lasagne.layers.get_all_param_values(network)))
        print('epoch_time ', (time.time() - time_epoch) / 60., 'minutes')

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
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
    main()
