import numpy as np
from skimage.util.shape import view_as_blocks
import skimage


class Layer(object):

    def __init__(self, receptive_field, dimensions):
        """ A layer contains a receptive field that is convolved over the input image"""
        self.receptive_field = receptive_field
        self.layer_above = None
        self.layer_below = None
        self.dimensions = dimensions  # The number of receptive fields in each dimension

    def accept_input(self, signal, learn=True):
        """ Takes an input and learns it, passes it to the next layer"""

        if np.shape(signal)[0] != np.shape(signal)[1]:
            raise Exception('Input signal must be square.')

        # Adds padding to the image
        padding_needed = int(len(signal[0]) % self.receptive_field.filter_shape[0])
        signal = skimage.util.pad(signal, [padding_needed, padding_needed], 'minimum')

        # Split the input up into patches
        patches = view_as_blocks(signal, self.receptive_field.filter_shape)
        # WARNING: This IS PROBABLY be wrong
        patches = patches.reshape(-1, self.receptive_field.filter_shape[0], self.receptive_field.filter_shape[0])

        # Send those patches to the receptive field, and get the filter that was most strongly excited
        # for each patch
        excited_filters = []
        for patch in patches:
            cfilter = self.receptive_field.accept_input(patch, learn=learn)
            excited_filters.append(cfilter)

        if self.layer_below:
            # Now we know the filter that was excited for each receptive field in the input image
            # Create a new 'image' where each pixel contains a 2d position value, this value corresponds to the
            # position of the excited filter.
            positions = [cfilter.position for cfilter in excited_filters]
            positions = np.reshape(positions, self.receptive_field.shape)
            self.layer_below.accept_input(positions, learn=learn)

            # WARNING!!!!
            # TODO  Check that the reshape is working in the correct order

    def visualize(self, upstream=None):
        if self.layer_above is None:
            self.receptive_field.visualize()
        if self.layer_below:
            self.layer_below.visualize(self.receptive_field)
