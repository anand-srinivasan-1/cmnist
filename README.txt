This repository is a simple neural network implementation in C that learns to recognize images in the MNIST dataset (handwritten digits).

It is implemented by creating a tree, which itself is a type of directed acyclic graph, to represent the matrix computations. There is a function that evaluates the network by evaluating each node's input nodes before evaluating the node itself, and another function for recursively traversing the network to compute the derivative of the error with respect to each parameter. With a 3 layer network, the model reaches ~90% accuracy after the first training epoch, and continues to become more accurate from there, eventually reaching ~96% accuracy.

This demonstrates that sometimes adding a bit of abstraction and complexity is a good thing even for small projects. This project was implemented in a few hours after using the graph (technically tree) model of computation, which was much faster than manually specifying every computation in a procedural style.

The decision to use a tree was because that would make it much easier to recursively traverse the computation graph to free memory.
