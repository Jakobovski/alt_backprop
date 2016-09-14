import numpy as np
from skimage.util.shape import view_as_blocks
import skimage


class Layer(object):

    def __init__(self, receptive_field):
        """ A layer contains a receptive field that is convolued over the input image"""
        self.receptive_field = receptive_field
        self.layer_above = None
        self.layer_below = None

    def accept_input(self, signal, learn=True):
        """ Takes an input and learns it, passes it to the next layer"""

        if np.shape(signal)[0] != np.shape(signal)[1]:
            raise Exception('Input signal must be square.')

        # Adds padding to the image
        padding_needed = int(len(signal[0]) % self.receptive_field.neuron_shape[0])
        signal = skimage.util.pad(signal, [padding_needed, padding_needed], 'minimum')

        # Split the input up into patches
        patches = view_as_blocks(signal, self.receptive_field.neuron_shape)
        # WARNING!!!!
        # TODO: This IS PROBABLY be wrong
        patches = patches.reshape(-1, 6, 6)

        # Send those patches to the receptive field, and get the neuron that was excited
        excited_neurons = []
        for patch in patches:
            neuron = self.receptive_field.accept_input(patch, learn=learn)
            excited_neurons.append(neuron)


        if self.layer_below:
            print len(excited_neurons)
            # Now we know the neuron that was excited for each receptive field in the input image
            # Create a new 'image' where each pixel contains a 2d position value, this value corresponds to the
            # position of the excited neuron.
            output = np.reshape(excited_neurons, self.receptive_field.neuron_shape)
            self.layer_below.accept_input(output, learn=learn)

            # WARNING!!!!
            # Check that the reshape is working in the correct order