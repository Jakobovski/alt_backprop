from __future__ import division
import tensorflow.examples.tutorials.mnist.input_data as input_data
import alt_backprop.config as config
from alt_backprop.models.layer import Layer
from alt_backprop.utils.patches import plot_patches
import numpy as np
from skimage.util.shape import view_as_blocks
import skimage

if __name__ == '__main__':
    # # Number of images to sample from
    SAMPLE_IMAGES = 2000
    mnist = input_data.read_data_sets('MNIST_data')
    images, labels = mnist.train.next_batch(SAMPLE_IMAGES)

    # Initialize layer 1
    layer_1 = Layer((28, 28), 11**2, (4, 4))

    for idx, image in enumerate(images):
        image = image.reshape(28, 28)
        layer_1.accept_input(image, learn=True)

        if idx % 10 == 0:
            print 'Percent: {0:0.1f}, LR: {0:0.3f}'.format(idx * 100 / len(images), config.LEARNING_RATE)
            config.LEARNING_RATE *= .995


    import pickle
    output = open('layer_1_4x4', 'wb')
    pickle.dump(layer_1, output)
    output.close()

    # # Display the results
    layer_1.visualize()
    input('Press ENTER to exit')

    # ============================
    # Recreate original from neuron weights (Generative mode)
    # ============================
    # for image in images[0:5]:
    #     image = image.reshape(28, 28)
    #     padding_needed = int(len(image[0]) % 4)
    #     image = skimage.util.pad(image, [padding_needed, padding_needed], 'minimum')

    #     patches = view_as_blocks(image, (4, 4))
    #     patches = patches.reshape(-1, 4, 4)

    #     recreation_patches = np.ones(patches.shape)
    #     for idx, patch in enumerate(patches):
    #         cneuron = layer_1.receptive_field.accept_input(patch, learn=False)
    #         recreation_patches[idx] = cneuron.weights
    #     plot_patches(recreation_patches)


    # input('Press ENTER to exit')
