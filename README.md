# Physics-informed Gaussian Processes for Topology Optimization

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Code for the paper [Simultaneous and Meshfree Topology Optimization with Physics-informed Gaussian Processes
](https://arxiv.org/abs/2408.03490), where we develop a new class of TO methods based on the framework of Gaussian processes (GPs) whose mean functions are parameterized via deep neural networks. Specifically, we place GP priors on all design and state variables to represent them via parameterized continuous functions. These GPs share a deep neural network as their mean function but have as many independent kernels as there are state and design variables. We estimate all the parameters of our model in a single for loop that optimizes a penalized version of the performance metric where the penalty terms correspond to the state equations and design constraints.


