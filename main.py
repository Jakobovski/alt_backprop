from __future__ import division
import tensorflow.examples.tutorials.mnist.input_data as input_data

from receptive_field import ReceptiveField
import utils
import config

if __name__ == '__main__':
    # Number of images to sample from
    SAMPLE_IMAGES = 500

    rec_field = ReceptiveField(121, 36)
    mnist = input_data.read_data_sets('MNIST_data')

    images, labels = mnist.train.next_batch(SAMPLE_IMAGES)
    patches = utils.extract_patches(images, (6, 6))

    for idx, patch in enumerate(patches):
        rec_field.learn(patch)

        if idx % 100 == 0:
            print 'Percent:', idx * 100 / len(patches), 'LR:', config.LEARNING_RATE
            config.LEARNING_RATE *= .999

    # Display the results
    utils.show_receptive_field(rec_field)
