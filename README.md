Micrograd Implemented in C++

This project is an implementation of the micrograd automatic differentiation
It makes use of share_ptr smart pointers to manage computation graphs with multiple parent dependencies.
This alllowed recursive back propagation to be carried out.

The project then goes on to implement different optimisers, including ADAM and RMSprop.
Finally, BCEloss is implemented alongside OpenMP parallelisation.
The entire framework was tested on a moons dataset.


