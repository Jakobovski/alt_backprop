from receptive_field import ReceptiveField


class Layer(object):

    def __init__(self):
        """ A layer contains a receptive field that is convolued over the input image"""
        self.receptive_field = ReceptiveField(121, 36)
        self.layer_above = None
        self.layer_below = None

    def accept_input(self, input, learn=True):
        """ Takes an input and learns it, passes it to the next layer"""
        
        # Split the input up into patches
        patches = []

        # Send those patches to the receptive field, and get the neuron that was excited
        excited_neurons = []
        for patch in patches
            neuron = rec_field.accept_input(patch, learn=learn)
            excited_neurons.append(neuron)

        # Now we know the neuron that was excited for each receptive field in the input image
        # Create a new 'image' where each pixel contains a 2d position value, this value corresponds to the
        # position of the excited neuron. 
        new_shape =  (self.rec_field.side_len, self.rec_field.side_len)
        output = np.reshape([neuron.position for neuron in excited_neurons], new_shape)
        self.layer_below.accept_input(output, learn=learn)
        
        # neuron will take positions as input and move toward them. 

        pass