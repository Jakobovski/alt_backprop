from __future__ import division
import math
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from alt_backprop.models.neuron import Neuron
import alt_backprop.config as cfg
matplotlib.use('TkAgg')


class ReceptiveField2D(object):

    def __init__(self, num_neurons, input_dims):
        """A receptive field that excepts 1D values for each input pixel. This is best used for grayscale images, color
            images will use a 3D receptive field (rgb)

        This represents a small region of input space. A RF consists of many neurons, these neurons
        compete with each other to learn representations.

        It can be thought of as a group of neurons.
        """
        self.num_neurons = num_neurons

        # The neurons need to be a square
        self.side_len = int(math.sqrt(num_neurons))
        self.shape = (self.side_len, self.side_len)
        assert math.sqrt(self.num_neurons) % 1 == 0

        # This is the shape of the image that is input into the receptive field
        self.neuron_shape = input_dims

        # Setup the dict to hold the neurons
        self.neurons = {}

        # Initialize the neuron
        for r_idx in range(self.side_len):
            for c_idx in range(self.side_len):
                position = (r_idx, c_idx)
                self.neurons[position] = Neuron(position, self.neuron_shape, self)

    def accept_input(self, signal, learn):
        for pos, conv_neuron in self.neurons.iteritems():
            conv_neuron.get_excitation(signal)

        max_neuron = max(self.neurons.values(), key=lambda cneuron: cneuron.excitation)
        if learn:
            self._run_learning(max_neuron, signal)

        return max_neuron

    def _run_learning(self, max_neuron, signal):
        # Get the neuron with the greatest excitation
        max_neuron.move_towards(signal, cfg.LEARNING_RATE)  # TODO. We might have double learning here.. Thats probably OK

        # Find the neighbors of the recently fired neurons and make them learn a bit
        for neighbor in max_neuron.get_neighbor_cords(1, cfg.POS_NEIGHBOR_MAX):
            dx = abs(max_neuron.position[0] - neighbor[0])
            dy = abs(max_neuron.position[1] - neighbor[1])
            learning_rate = cfg.NEIGHBOR_LEARNING_RATE / max(dy, dx)**2
            self.neurons[neighbor].move_towards(signal, learning_rate=learning_rate)

    def visualize(self):
        """Displays an image of a receptive field"""
        dneuron = self.neuron_shape[0]
        d = int(math.sqrt(self.num_neurons)) * self.neuron_shape[0]
        large_image = np.ones((d, d))

        for pos, cneuron in self.neurons.iteritems():
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
