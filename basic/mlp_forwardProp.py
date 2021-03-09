# implementing a multi layer perceptron 

import numpy as np

class MLP:
    def __init__(self, n_inputs=3, n_hidden=[3,3], n_outputs=2):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        # total layers present in a this MLP
        # LOL - List of lists
        layers = [self.n_inputs] + self.n_hidden + [self.n_outputs]

        # initiating random weights matrix
        weights = []
        for i in range(len(layers)-1):
            curr_weights = np.random.rand(layers[i],layers[i+1])
            weights.append(curr_weights)
        self.weights = (weights)
        

        # forward propagation method
    def forward_prop(self, inputs):
        # calculating net inputs / weighted sum
        for w in self.weights:
            net_inputs = np.dot(inputs,w)
            # calculating activation
            activations = self._sigmoid(net_inputs)
        return activations

    def _sigmoid(self, net_inputs):
        return 1 / 1 + np.exp(net_inputs)


if __name__ == "__main__":

    # creating MLP network / Neural network
    mlp = MLP()  # using default values of MLP for now

    # creating random inputs
    inputs = np.random.rand(mlp.n_inputs)  # n_inputs = n_neurons

    # forward propagation

    output = mlp.forward_prop(inputs)

    print("The inputs given to MLP are {}\n", format(inputs))
    print("The output given from the MLP layer is {}\n", format(output))