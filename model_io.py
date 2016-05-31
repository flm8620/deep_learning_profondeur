from __future__ import print_function

import numpy as np

import lasagne
import cifar10_nin
import lenet5


def load_model(model_type, model_file, nOutput, input_var):
    if model_type == 'lenet':
        net = lenet5.build_lenet5(input_var, nOutput)
        net_output = net['output']
    elif model_type == 'cifar':
        net = cifar10_nin.build_model2(input_var, nOutput)
        net_output = net['output']
    else:
        print("Unrecognized model type %r." % model_type)
        return
    if model_file is not None:
        with np.load(model_file) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net_output, param_values)

    return net, net_output


def save_model(model_file, net_output):
    np.savez(model_file, *(lasagne.layers.get_all_param_values(net_output)))
