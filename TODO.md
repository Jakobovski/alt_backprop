## TODO

MNIST Setup
0: Input 28x28  (784)
1. 4x4 span, with 12x12 capacity (outputs a 7x7x2 representations)  (98 total)
2. 2x2 span, with 12x12 capacity (uses buffer of 1) (outputs 4x4x2 representation) (32 total)
3. Send to 2-layer linear classifier


Each RF has a span (how many pixels in the layer above it receives input from) and a capacity, the number of representations it containts



New naming scheme
- receptive field
- filter => neuron

== Map growth ==
Maybe allow the maps to grows? 


=== Performance Related ==
- Test multi-threading the learning of filters. See if this improves performance. It probably wont because of the overhead of managing threads



do i need some type of dropout?

http://www.cell.com/neuron/fulltext/S0896-6273(17)30509-3?utm_content=buffer1aad4&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

(On continual learning)These empirical insights are consistent with theoretical models that suggest that memories can be protected from interference through synapses that transition between a cascade of states with different levels of plasticity 

pass  “characters challenge” (Lake et al., 2016). 