from __future__ import print_function

import argparse

import numpy as np

import theano
import theano.tensor as T
import lasagne

import load_data
import model_io
import six.moves.cPickle as pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name", choices=['cifar', 'lenet'])
    parser.add_argument("model_file", help="model file")
    parser.add_argument('layer', help='layer name to get output')
    parser.add_argument('--no-separate', help='split the data', action='store_true')
    parser.add_argument('--first-part', help='take first part of data instead of the second', action='store_true')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-n', '--data-num', type=int)

    args = parser.parse_args()

    model = args.model
    batch_size = args.batch_size
    separate = not args.no_separate
    model_file = args.model_file
    layer_name = args.layer
    load_first_part = args.first_part
    data_num = args.data_num

    filename = model + '_' + layer_name + '_output.save'
    print('--Parameters--')
    print('  model         : ', model)
    print('  layer name    : ', layer_name)
    print('  batch_size    : ', batch_size)
    print('  model_file    : ', model_file)
    print('  middle output will be saved to : ', filename)
    print('  separate data :', separate)
    if separate:
        print('    take first or second part of data :', 'first' if load_first_part else 'second')
    print('batch_size=', batch_size)

    if separate:
        nOutput = 5
    else:
        nOutput = 10

    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data.load_dataset(model, separate, load_first_part)
    if data_num:
        X_train = X_train[:data_num]
        y_train = y_train[:data_num]
        X_val = X_val[:data_num]
        y_val = y_val[:data_num]
        X_test = X_test[:data_num]
        y_test = y_test[:data_num]

    print(len(X_train), 'train images')
    print(len(X_val), 'val images')
    print(len(X_test), 'test images')

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    net, net_output = model_io.load_model(model, model_file, nOutput, input_var)

    # middle_output = theano.function([input_var], net[layer_name].output)
    print("Getting middle output...")

    output = lasagne.layers.get_output(net[layer_name])
    get_output = theano.function([input_var], output.flatten(2))

    output_shape = np.array(lasagne.layers.get_output_shape(net[layer_name]))
    print('layer ' + layer_name + ' shape :', output_shape)

    all_train_output = []
    all_train_y = []
    all_test_output = []
    all_test_y = []
    print('getting from train')
    for batch in load_data.iterate_minibatches(X_train, y_train, batch_size, shuffle=False):
        print('.', end='', flush=True)
        inputs, targets = batch
        batch_output = get_output(inputs)  # a numpy ndarray
        all_train_output.extend(batch_output.tolist())
        all_train_y.extend(targets.tolist())
    print()
    print('getting from test')
    for batch in load_data.iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
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

    with open(filename, 'wb') as f:
        pickle.dump([all_train_output, all_train_y, all_test_output, all_test_y], f, protocol=pickle.HIGHEST_PROTOCOL)
    print('... saved to ', filename)


if __name__ == '__main__':
    main()
