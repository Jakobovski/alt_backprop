from __future__ import division
import math

from neuron import Neuron
import config as cfg


class ReceptiveField(object):

    def __init__(self, num_neurons, neuron_size):
        """A receptive field represents a small region of input space. A RF consists of many Neurons, these neurons
        compete with each other to learn representations.
        """
        self.num_neurons = num_neurons
        self.neuron_size = neuron_size

        # The neurons need to be a square
        self.side_len = int(math.sqrt(num_neurons))
        assert math.sqrt(self.num_neurons) % 1 == 0

        # Setup the dict to hold the neurons
        self.neurons = {}

        # Initialize the neuron
        for r_idx in range(self.side_len):
            for c_idx in range(self.side_len):
                position = (r_idx, c_idx)
                self.neurons[position] = Neuron(position, neuron_size, self)

    def accept_input(self, signal, learn):
        for pos, conv_neuron in self.neurons.iteritems():
            conv_neuron.get_excitation(signal)

        max_neuron = max(self.neurons.values(), key=lambda neuron: neuron.excitation)
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

