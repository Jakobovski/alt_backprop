from __future__ import division
import math
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from alt_backprop.filter import Filter
import config as cfg
matplotlib.use('TkAgg')


class ReceptiveField(object):

    def __init__(self, num_filters, filter_size):
        """A receptive field represents a small region of input space. A RF consists of many filters, these filters
        compete with each other to learn representations.

        It can be thought of as a group of filters.
        """
        self.num_filters = num_filters

        # The filters need to be a square
        self.side_len = int(math.sqrt(num_filters))
        assert math.sqrt(self.num_filters) % 1 == 0
        self.shape = (self.side_len, self.side_len)

        # This is the shape of the image that is input into the receptive field
        self.filter_shape = (math.sqrt(filter_size), math.sqrt(filter_size))

        # Setup the dict to hold the filters
        self.filters = {}

        # Initialize the filter
        for r_idx in range(self.side_len):
            for c_idx in range(self.side_len):
                position = (r_idx, c_idx)
                self.filters[position] = Filter(position, self.filter_shape, self)

    def accept_input(self, signal, learn):
        for pos, conv_filter in self.filters.iteritems():
            conv_filter.get_excitation(signal)

        max_filter = max(self.filters.values(), key=lambda cfilter: cfilter.excitation)
        if learn:
            self._run_learning(max_filter, signal)

        return max_filter

    def _run_learning(self, max_filter, signal):
        # Get the filter with the greatest excitation
        max_filter.move_towards(signal, cfg.LEARNING_RATE)  # TODO. We might have double learning here.. Thats probably OK

        # Find the neighbors of the recently fired filters and make them learn a bit
        for neighbor in max_filter.get_neighbor_cords(1, cfg.POS_NEIGHBOR_MAX):
            dx = abs(max_filter.position[0] - neighbor[0])
            dy = abs(max_filter.position[1] - neighbor[1])
            learning_rate = cfg.NEIGHBOR_LEARNING_RATE / max(dy, dx)**2
            self.filters[neighbor].move_towards(signal, learning_rate=learning_rate)

    def visualize(self):
        """Displays an image of a receptive field"""
        dfilter = self.filter_shape[0]
        d = int(math.sqrt(self.num_filters)) * self.filter_shape[0]
        large_image = np.ones((d, d))

        for pos, cfilter in self.filters.iteritems():
            img = cfilter.weights
            offset = [pos[0] * dfilter, pos[1] * dfilter]
            large_image[offset[0]:offset[0] + dfilter, offset[1]:offset[1] + dfilter] = img

        fig = plt.figure()
        plt.imshow(large_image, cmap='Greys_r', interpolation='none')
        plt.show(block=False)

        # Make the window on top
        if matplotlib.get_backend() == 'TkAgg':
            fig.canvas.manager.window.attributes('-topmost', 1)
        else:
            fig.window.raise_()
