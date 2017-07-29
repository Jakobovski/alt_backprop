from __future__ import division
import pickle

import tensorflow.examples.tutorials.mnist.input_data as input_data

from alt_backprop.utils import utils, patch_util

import alt_backprop.config as config
from alt_backprop.models.receptive_field_1d import ReceptiveField1D
from alt_backprop.models.receptive_field_2d import ReceptiveField2D
from alt_backprop.models.layer import Layer
from alt_backprop.utils.patch_util import plot_patches

if __name__ == '__main__':
    # # =========== Initialize layer 1 ===========
    # l1_rec_field = ReceptiveField1D(144, (4,4), None)
    # layer_1 = Layer(l1_rec_field, (28, 28))
    #
    # # =========== Initialize layer 2 ===========
    # l2_rec_field = ReceptiveField2D(20*20, (2,2,2), l1_rec_field)
    # layer_2 = Layer(l2_rec_field, (7,7))
    #
    # layer_1.set_output_layer(layer_2)
    # layer_2.set_input_layer(layer_1)
    #
    # # =========== Prepare data ===========
    # TRAINING_SAMPLE_SIZE = 3500
    # mnist = input_data.read_data_sets('MNIST_data')
    # images, labels = mnist.train.next_batch(TRAINING_SAMPLE_SIZE)
    #
    # # =========== Start training ===========
    # for idx, image in enumerate(images):
    #     image = image.reshape(28, 28)
    #     layer_1.accept_input(image, learn=True)
    #
    #     if idx % 20 == 0:
    #         print 'Percent: {}, LR: {}'.format(idx * 100 / len(images), config.LEARNING_RATE)
    #         config.LEARNING_RATE *= .999
    #
    #         if idx == 350:
    #             layer_1.set_output_layer(layer_2)
    #             layer_2.set_input_layer(layer_1)
    #
    #
    # output = open('layer_1_144x4x4', 'wb')
    # pickle.dump(layer_1, output)
    # output.close()
    #
    # output = open('layer_2_144x2x2x2', 'wb')
    # pickle.dump(layer_2, output)
    # output.close()
    #
    # # Display the results
    # layer_1.visualize()
    # input('Press ENTER to exit')
    #
    #
    #
    # # ============================
    # # Recreate original from neuron weights (Generative mode)
    # # ============================
    import numpy as np
    import pickle
    from skimage.util.shape import view_as_blocks

    TRAINING_SAMPLE_SIZE = 10
    mnist = input_data.read_data_sets('MNIST_data')
    images, labels = mnist.train.next_batch(TRAINING_SAMPLE_SIZE)

    output = open('layer_1_144x4x4', 'rb')
    layer_1 = pickle.load(output)
    output.close()

    output = open('layer_2_144x2x2x2', 'rb')
    layer_2 = pickle.load(output)
    output.close()

    # layer_1.visualize()
    # input('Press ENTER to exit')

    for image in images[1:2]:
        image = image.reshape(28, 28)
        image = np.zeros((28,28))
        # patches = patch_util.extract_patches(image,(4,4))
        img = layer_1.recreate_input(image)
        # print img.shape
        utils.plot_image(img)

        # utils.plot_image(img)
        # recreation_patches = np.ones((128, 4, 4))
        # count = 0
        # excited_l2_subfields = layer_1.accept_input(image, learn=False)
        # for sf in excited_l2_subfields:
        #     for row in sf.weights:
        #         for w in row:
        #             img = layer_1.receptive_field.get_image_representation(w)
        #             recreation_patches[count] = img
        #             count += 1
        #             print count
        # plot_patches(recreation_patches)
        # #
        # patches = view_as_blocks(image, (4, 4))
        # patches = patches.reshape(-1, 4, 4)
        #
        # recreation_patches = np.ones((4*4, 8, 8))
        # count = 0
        # for idx, patch in enumerate(patches):
        #     excited_l2_neurons = layer_1.accept_input(patch, learn=False)
        #     for n in excited_l2_neurons:
        #         img = layer_1.receptive_field.get_image_representation(n.weight)
        #         recreation_patches[count] = img
        #         count += 1
        #
        # plot_patches(recreation_patches)

    input('Press ENTER to exit')
