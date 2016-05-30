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
import random

import numpy as np
import theano
import theano.tensor as T

import lasagne
import cifar10_nin
from lenet5 import *
from load_data import *

from itertools import chain
import functools


def get_all_params_from_layers(layers, unwrap_shared=True, **tags):
    params = chain.from_iterable(l.get_params(
        unwrap_shared=unwrap_shared, **tags) for l in layers)
    return lasagne.utils.unique(params)


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def log_and_print(text, logfile):
    with open(logfile, 'a') as f:
        f.write(text)
        print(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name", choices=['cifar', 'lenet'])
    parser.add_argument('-n', '--num-epochs', type=int, default=20)
    parser.add_argument('-f', '--model-file', help="model file")
    parser.add_argument('--no-separate', help='split the data', action='store_true')
    parser.add_argument('--second-part', help='take second part of data instead of the first', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-l', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-t', '--test-only', action='store_true')
    parser.add_argument('-T', '--train-from-layer', help='only train on this layer and those layers after it, \
    don\'t update weights of layers before this layer')
    parser.add_argument('-p', '--prefix', help='prefix to add at the beginning of model save file')

    args = parser.parse_args()

    model = args.model
    batch_size = args.batch_size
    separate = not args.no_separate
    model_file = args.model_file
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    save_file_name = model + '_model'
    test_only = args.test_only
    load_first_part = not args.second_part
    train_from_layer = args.train_from_layer
    prefix = args.prefix

    if test_only and not model_file:
        print('you need to specify a model file to test')
        exit()

    if separate:
        save_file_name = 'half_' + save_file_name
        nOutput = 5
    else:
        nOutput = 10

    if train_from_layer:
        save_file_name = 'from_' + train_from_layer + save_file_name

    if prefix:
        save_file_name = prefix + save_file_name
    else:
        save_file_name = str(random.randint(10000, 99999)) + save_file_name

    logfile = save_file_name + '_log.txt'
    log_print = functools.partial(log_and_print, logfile=logfile)
    log_print('--Parameter--')
    log_print('  model={}'.format(model))
    log_print('  batch_size={}'.format(batch_size))
    log_print('  num_epochs={}'.format(num_epochs))
    log_print('  learning_rate={}'.format(learning_rate))
    log_print('  separate data :{}'.format(separate))
    if separate:
        s = '    take first or second part of data :'+('first' if load_first_part else 'second')
        log_print(s)
    log_print('  model_file :{}'.format(model_file))
    log_print('  nOutput = {}'.format(nOutput))
    log_print('  model will be saved to : {}'.format(save_file_name + '*.npz'))
    log_print('  log will be saved to : {}'.format(logfile))
    log_print('  test only :{}'.format(test_only))
    log_print('  only train from this layer : {}'.format(train_from_layer))
    log_print('  prefix to save file : {}'.format(prefix))

    log_print('')

    log_print("Loading data...")
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

    log_print('{} train images'.format(len(X_train)))
    log_print('{} val images'.format(len(X_val)))
    log_print('{} test images'.format(len(X_test)))

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    log_print("Building model and compiling functions...")
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

    if train_from_layer:
        layers_to_train = lasagne.layers.get_all_layers(network, treat_as_input=[net[train_from_layer]])
        params = get_all_params_from_layers(layers_to_train, trainable=True)
    else:
        params = lasagne.layers.get_all_params(network, trainable=True)

    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    if not test_only:
        log_print("Starting training...")

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
            log1 = "Epoch {} of {} took {:.3f}m".format(epoch + 1, num_epochs, (time.time() - start_time) / 60.)
            log2 = "  training loss:\t\t{:.6f}".format(train_err / train_batches)
            log3 = "  validation loss:\t\t{:.6f}".format(val_err / val_batches)
            log4 = "  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100)
            log_print(log1)
            log_print(log2)
            log_print(log3)
            log_print(log4)

            # Optionally, you could now dump the network weights to a file like this:

            model_file = save_file_name + str(epoch) + '.npz'
            log_print('model saved to ' + model_file)
            np.savez(model_file, *(lasagne.layers.get_all_param_values(network)))

    log_print('testing network ...')
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
    log_print("Final results:")
    log_print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    log_print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    main()
