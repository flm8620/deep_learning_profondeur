# Network in Network CIFAR10 Model
# Original source: https://gist.github.com/mavenlin/e56253735ef32c3c296d
# License: unknown

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/cifar10/model.pkl

from lasagne.layers import InputLayer, DropoutLayer, FlattenLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer

import lasagne


def build_model(input_var=None):
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False,
                             W=lasagne.init.GlorotUniform())
    net['cccp1'] = ConvLayer(
        net['conv1'], num_filters=160, filter_size=1, flip_filters=False)
    net['cccp2'] = ConvLayer(
        net['cccp1'], num_filters=96, filter_size=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['cccp2'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['drop3'] = DropoutLayer(net['pool1'], p=0.5)
    net['conv2'] = ConvLayer(net['drop3'],
                             num_filters=192,
                             filter_size=5,
                             pad=2,
                             flip_filters=False)
    net['cccp3'] = ConvLayer(
        net['conv2'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp4'] = ConvLayer(
        net['cccp3'], num_filters=192, filter_size=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['cccp4'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['drop6'] = DropoutLayer(net['pool2'], p=0.5)
    net['conv3'] = ConvLayer(net['drop6'],
                             num_filters=192,
                             filter_size=3,
                             pad=1,
                             flip_filters=False)
    net['cccp5'] = ConvLayer(
        net['conv3'], num_filters=192, filter_size=1, flip_filters=False)
    net['cccp6'] = ConvLayer(
        net['cccp5'], num_filters=10, filter_size=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['cccp6'],
                             pool_size=8,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['output'] = FlattenLayer(net['pool3'])

    return net


def build_model2(input_var=None):
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=32,
                             filter_size=5,
                             pad=2,
                             flip_filters=False,
                             W=lasagne.init.GlorotUniform(),
                             nonlinearity=lasagne.nonlinearities.rectify)
    net['pool1'] = PoolLayer(net['conv1'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['norm1'] = lasagne.layers.LocalResponseNormalization2DLayer(net['pool1'], n=3, alpha=5e-5)
    net['conv2'] = ConvLayer(net['norm1'],
                             num_filters=32,
                             filter_size=5,
                             pad=2,
                             flip_filters=False,
                             W=lasagne.init.GlorotUniform(),
                             nonlinearity=lasagne.nonlinearities.rectify)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['norm2'] = lasagne.layers.LocalResponseNormalization2DLayer(net['pool2'], n=3, alpha=5e-5)

    net['conv3'] = ConvLayer(net['norm2'],
                             num_filters=64,
                             filter_size=5,
                             pad=2,
                             flip_filters=False,
                             W=lasagne.init.GlorotUniform(),
                             nonlinearity=lasagne.nonlinearities.rectify)
    net['pool3'] = PoolLayer(net['conv3'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['output'] = lasagne.layers.DenseLayer(
        net['pool3'],
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return net

def build_quick(input_var=None):
    net = {}
    net['input'] = InputLayer((None, 3, 32, 32), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=32,
                             filter_size=5,
                             pad=2,
                             flip_filters=False,
                             W=lasagne.init.GlorotUniform(),
                             nonlinearity=lasagne.nonlinearities.rectify)
    net['pool1'] = PoolLayer(net['conv1'],
                             pool_size=3,
                             stride=2,
                             mode='max',
                             ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'],
                             num_filters=32,
                             filter_size=5,
                             pad=2,
                             flip_filters=False,
                             W=lasagne.init.GlorotUniform(),
                             nonlinearity=lasagne.nonlinearities.rectify)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)

    net['conv3'] = ConvLayer(net['pool2'],
                             num_filters=64,
                             filter_size=5,
                             pad=2,
                             flip_filters=False,
                             W=lasagne.init.GlorotUniform(),
                             nonlinearity=lasagne.nonlinearities.rectify)
    net['pool3'] = PoolLayer(net['conv3'],
                             pool_size=3,
                             stride=2,
                             mode='average_exc_pad',
                             ignore_border=False)
    net['output'] = lasagne.layers.DenseLayer(
        net['pool3'],
        num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return net