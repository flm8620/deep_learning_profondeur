from __future__ import print_function

import argparse

import numpy as np

import theano
import theano.tensor as T
import lasagne

import load_data
import model_io
import random

try:
    import PIL.Image as Image
except ImportError:
    import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name", choices=['cifar', 'lenet'])
    parser.add_argument("model_file", help="model file")
    parser.add_argument('layer', help='layer name to get image output')
    parser.add_argument('imageID', help='ID of image for input', type=int)
    parser.add_argument('-d', '--dataset', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--no-separate', help='split the data', action='store_true')
    parser.add_argument('--first-part', help='take first part of data instead of the second', action='store_true')

    args = parser.parse_args()

    model = args.model
    batch_size = 1
    separate = not args.no_separate
    model_file = args.model_file
    layer_name = args.layer
    chosen_set = args.dataset
    load_first_part = args.first_part
    imageID = args.imageID
    filename = str(random.randint(10000, 99999)) + '_' + model + '_' + layer_name + '_output.png'
    print('--Parameters--')
    print('  model         : ', model)
    print('  layer name    : ', layer_name)
    print('  batch_size    : ', batch_size)
    print('  model_file    : ', model_file)
    print('  middle output images will be saved to : ', filename)
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
    get_output_image = theano.function([input_var], output.flatten(3))

    output_shape = np.array(lasagne.layers.get_output_shape(net[layer_name]))
    foo, nKernel, h, w = output_shape
    print('layer ' + layer_name + ' shape :', output_shape)

    print('getting from' + chosen_set)
    if chosen_set == 'train':
        X_set = X_train
        y_set = y_train
    elif chosen_set == 'val':
        X_set = X_val
        y_set = y_val
    else:
        X_set = X_test
        y_set = y_test

    batch_output = get_output_image(np.array([X_set[imageID]]))
    images_output = batch_output[0]
    prediction = lasagne.layers.get_output(net_output)

    get_pred = theano.function([input_var], prediction)
    pred = get_pred(np.array([X_set[imageID]]))

    width = 1
    while width * width < nKernel:
        width += 1

    if width * width > nKernel:
        images_output = np.concatenate((images_output, np.zeros((width * width - nKernel, w * h))), axis=0)

    image = Image.fromarray(tile_raster_images(
        X=images_output,  # chose batch 0
        img_shape=(h, w), tile_shape=(width, width),
        tile_spacing=(1, 1)))
    image.save(filename)


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
        ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


if __name__ == '__main__':
    main()
