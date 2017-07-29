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
    l1_rec_field = ReceptiveField1D(144, (4,4), None)
    layer_1 = Layer(l1_rec_field, (28, 28), name='l1')

    # =========== Initialize layer 2 ===========
    l2_rec_field = ReceptiveField2D(20*20, (2,2,2), l1_rec_field)
    layer_2 = Layer(l2_rec_field, (7,7),  name='l2')

    # =========== Prepare data ===========
    TRAINING_SAMPLE_SIZE = 3000
    mnist = input_data.read_data_sets('MNIST_data')
    images, labels = mnist.train.next_batch(TRAINING_SAMPLE_SIZE)

    # =========== Start training ===========
    for idx, image in enumerate(images):
        image = image.reshape(28, 28)
        layer_1.accept_input(image, learn=True)

        if idx % 10 == 0:
            print 'Percent: {}, LR: {}'.format(idx * 100 / len(images), config.LEARNING_RATE)
            config.LEARNING_RATE *= .995

        if idx == 20:
            layer_1.set_output_layer(layer_2)
            layer_2.set_input_layer(layer_1)

    output = open('layer_1_144x4x4', 'wb')
    pickle.dump(layer_1, output)
    output.close()

    output = open('layer_2_144x2x2x2', 'wb')
    pickle.dump(layer_2, output)
    output.close()

    # Display the results
    layer_1.visualize()
    input('Press ENTER to exit')
