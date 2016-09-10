from __future__ import division
import numpy as np


class ConvFilter(object):

    def __init__(self, position, size, layer):
        self.position = position
        self.size = size
        self.weights = np.random.uniform(0, 1.0, size)
        self.layer = layer

        # Pre and post inhibition excitation
        self.pre_inhib_excitation = 0
        self.post_inhib_excitation = 0

        # A dictionary to cache results form get_neighbor_cords()
        self._cache = {}

    def get_pre_inhib_excitation(self, image):
        self.pre_inhib_excitation = (1 / np.linalg.norm(self.weights - image))
        return self.pre_inhib_excitation

    def get_post_inhib_excitation(self):
        self.post_inhib_excitation = self.pre_inhib_excitation + self.last_inhibition + self.layer.threshold
        # self.reset_inhibition()
        return self.post_inhib_excitation

    def reset_inhibition(self):
        self.last_inhibition = 0

    def move_towards(self, image, learning_rate):
        """ Makes the filters weights move toward the passed image"""
        self.weights += (image - self.weights) * learning_rate

    def get_neighbor_cords(self, nmin, nmax):
        """
        nmin: The neighbors that are at least 1 distance from the position.
        nmax: the max distance to get neighbors, (not inclusive.).
        returns the coordinates of neighbors that are at least `min` distance, but less than `max` distance.
        """
        cache_val = self._cache.get((nmin, nmax), None)
        if cache_val:
            return cache_val

        side_length = self.layer.side_len
        i, j = self.position
        neighbors = []

        x_range = [i - nmax, i + nmax]
        y_range = [j - nmax, j + nmax]

        for x in range(x_range[0] + 1, x_range[1]):
            for y in range(y_range[0] + 1, y_range[1]):
                pos = (x, y)
                dx = abs(x - i)
                dy = abs(y - j)

                if dx < nmin and dy < nmin:
                    continue
                elif pos[0] < 0 or pos[0] > side_length - 1:
                    continue
                elif pos[1] < 0 or pos[1] > side_length - 1:
                    continue
                else:
                    neighbors.append(pos)

        self._cache[(nmin, nmax)] = neighbors
        return neighbors

    def inhibit(self, amount):
        """Inhibits this filter in proportion to the amount"""
        assert amount >= 0
        self.last_inhibition += amount
