from __future__ import division
import tensorflow.examples.tutorials.mnist.input_data as input_data

from alt_backprop.receptive_field import ReceptiveField
import alt_backprop.utils as utils
import alt_backprop.config as config
from alt_backprop.layer import Layer


if __name__ == '__main__':
    # Number of images to sample from
    SAMPLE_IMAGES = 40
    mnist = input_data.read_data_sets('MNIST_data')
    images, labels = mnist.train.next_batch(SAMPLE_IMAGES)

    # Initialize layer 1
    l1_rec_field = ReceptiveField(121, 36)
    layer_1 = Layer(l1_rec_field, (11, 11))

    # Initialize layer 2
    l2_rec_field = ReceptiveField(121, 9)
    layer_2 = Layer(l2_rec_field, ())

    layer_1.layer_above = layer_2
    layer_2.layer_below = layer_1

    for idx, image in enumerate(images):
        image = image.reshape(28, 28)
        layer_1.accept_input(image, learn=True)

        if idx % 10 == 0:
            print 'Percent:', idx * 100 / len(images), 'LR:', config.LEARNING_RATE
            config.LEARNING_RATE *= .999

    # Display the results
    utils.show_receptive_field(l1_rec_field)
    utils.show_receptive_field(l2_rec_field)
    input('Press ENTER to exit')
