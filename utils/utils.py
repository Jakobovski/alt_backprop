import math
import numpy as np
from sklearn.feature_extraction import image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def patches_3d_to_2d(patches):
    z, y, x = patches.shape
    return patches.reshape(z, x * y)


def extract_patches(images, patch_shape, unique=True):
    x, y = patch_shape
    all_patches = np.ndarray(shape=(0, x * y), dtype=float, order='F')

    for img in images:
        img = img.reshape(28, 28)

        # Add a row and column to the img to make it 30x30 so we can reconstruct it better using 6x6 or 5x5 neurons
        col = np.zeros((28, 2), dtype=float)
        img = np.append(img, col, axis=1)
        row = np.zeros((2, 30), dtype=float)
        img = np.append(img, row, axis=0)

        patches = patches_3d_to_2d(image.extract_patches_2d(img, (x, y)))
        all_patches = np.append(all_patches, patches, axis=0)

    return all_patches


def show_patches(images):
    """Accepts a a list of arrays that are images and displays them"""
    for img in images:
        d = int(len(img)**.5)
        img = img.reshape(d, d)
        plt.figure()
        plt.imshow(img, cmap='Greys_r', interpolation='none')
    plt.show()


def show_receptive_field(rec_field):
    """Displays an image of a receptive field"""
    dneuron = rec_field.neuron_shape[0]
    d = int(math.sqrt(rec_field.num_neurons)) * rec_field.neuron_shape[0]
    large_image = np.ones((d, d))

    for pos, cneuron in rec_field.neurons.iteritems():
        img = cneuron.weights
        offset = [pos[0] * dneuron, pos[1] * dneuron]
        large_image[offset[0]:offset[0] + dneuron, offset[1]:offset[1] + dneuron] = img

    fig = plt.figure()
    plt.imshow(large_image, cmap='Greys_r', interpolation='none')
    plt.show(block=False)

    # Make the window on top
    if matplotlib.get_backend() == 'TkAgg':
        fig.canvas.manager.window.attributes('-topmost', 1)
    else:
        fig.window.raise_()
