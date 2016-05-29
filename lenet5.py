from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


def build_lenet5(input_var=None):
    # Written by Leman FENG
    net = {}

    net['input'] = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                             input_var=input_var)

    net['conv1'] = lasagne.layers.Conv2DLayer(
        net['input'], num_filters=20, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'], pool_size=(2, 2))

    net['conv2'] = lasagne.layers.Conv2DLayer(
        net['pool1'], num_filters=50, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify)

    net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'], pool_size=(2, 2))

    net['full'] = lasagne.layers.DenseLayer(
        net['pool2'],
        num_units=500,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    net['output'] = lasagne.layers.DenseLayer(
        net['full'],
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return net['output']
