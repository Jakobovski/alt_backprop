from __future__ import division
import tensorflow.examples.tutorials.mnist.input_data as input_data

from alt_backprop.receptive_field import ReceptiveField
import alt_backprop.utils.utils as utils
import alt_backprop.config as config
from alt_backprop.layer import Layer


if __name__ == '__main__':
    # Number of images to sample from
    SAMPLE_IMAGES = 1000
    mnist = input_data.read_data_sets('MNIST_data')
    images, labels = mnist.train.next_batch(SAMPLE_IMAGES)

    # Initialize layer 1
    l1_rec_field = ReceptiveField(121, 36)
    layer_1 = Layer(l1_rec_field)

    for idx, image in enumerate(images):
        image = image.reshape(28, 28)
        layer_1.accept_input(image, learn=True)

        if idx % 10 == 0:
            print 'Percent:', idx * 100 / len(images), 'LR:', config.LEARNING_RATE
            config.LEARNING_RATE *= .99

    # Display the results
    utils.show_receptive_field(l1_rec_field)
    input('Press ENTER to exit')
