from __future__ import division
import math

from conv_filter import ConvFilter
import config as cfg


class ReceptiveField(object):

    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.threshold = -1
        self._thresh_count = 0  # A cheat to prevent infinitue recursion

        # The filters need to be a square
        self.side_len = int(math.sqrt(num_filters))
        assert math.sqrt(self.num_filters) % 1 == 0

        # Setup the dict to hold the filters
        self.filters = {}

        # Initialize the filter
        for r_idx in range(self.side_len):
            for c_idx in range(self.side_len):
                position = (r_idx, c_idx)
                self.filters[position] = ConvFilter(position, filter_size, self)

    def learn(self, image):
        """Accepts an image and runs it through the filters until its learned"""
        # Run the image through all the filters
        for pos, conv_filter in self.filters.iteritems():
            conv_filter.reset_inhibition()
            conv_filter.get_pre_inhib_excitation(image)

        # self._run_inhibition()
        # self.adjust_threshold()
        self._run_learning(image)

    def _run_learning(self, image):
        # Get the filter with the greatest excitation
        max_filter = max(self.filters.values(), key=lambda cfilter: cfilter.get_post_inhib_excitation())
        max_filter.move_towards(image, cfg.LEARNING_RATE)  # TODO. We might have double learning here.. Thats probably OK

        # Find the neighbors of the recently fired filters and make them learn a bit
        for neighbor in max_filter.get_neighbor_cords(1, cfg.POS_NEIGHBOR_MAX):
            dx = abs(max_filter.position[0] - neighbor[0])
            dy = abs(max_filter.position[1] - neighbor[1])
            learning_rate = cfg.NEIGHBOR_LEARNING_RATE / max(dy, dx)**2
            self.filters[neighbor].move_towards(image, learning_rate=learning_rate)

        # print cfilter.post_inhib_excitation
        # Find the neighbors of the recently fired filters and make them learn a bit
        # for neighbor in cfilter.get_neighbor_cords(1, cfg.POS_NEIGHBOR_MAX):
        # self.filters[neighbor].move_towards(image, learning_rate=cfg.NEIGHBOR_LEARNING_RATE)

    def _run_inhibition(self):
        # Find distant neighbors and inhibit by the excitation amount of this cfilter
        for pos, cfilter in self.filters.iteritems():
            for cord in cfilter.get_neighbor_cords(nmin=2, nmax=cfg.INHIBIT_NEIGHBOR_MAX):
                self.filters[cord].inhibit(cfilter.last_excitation)

    def adjust_threshold(self):
        """Reduces or increases the global threshold Dependant on the number of active neurons firing per
        Receptive field"""
        pass
        # num_fired = sum(1 if cfilter.get_excitation(re_multiply=False) else 0 for pos, cfilter in self.filters.iteritems())
        # print 'Num fired:', num_fired
        # if self._thresh_count > 20:
        #     self._thresh_count = 0
        #     return
        # else:
        #     self._thresh_count += 1

        # if num_fired > cfg.MAX_NUERONS_PER_EXCITATION:
        #     print 'Reducing threshold:', self.threshold
        #     self.threshold -= .1
        #     self.adjust_threshold()
        # elif num_fired < cfg.MIN_NUERONS_PER_EXCITATION:
        #     print 'Increasing threshold:', self.threshold
        #     self.threshold += .1
        #     self.adjust_threshold()
        # else:
        #     print 'Num fired:', num_fired
        #     self._thresh_count = 0
        #     return
