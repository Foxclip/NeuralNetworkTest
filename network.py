import numpy as np
import random
import math
import numba
from numba import float32
from numba import cuda

CLIP_VALUES = False             # limit weights and biases to 0.0..1.0
MAX_NEURONS = 100               # needed for memry allocation on CUDA


def sigmoid_exp(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


@cuda.jit(device=True)
def sigmoid_tanh(x):
    """
    same as sigmoid_exp, but calculated using tanh
    sigmoid_exp gives overflow error if weights are too high, but this function does not
    """
    return (math.tanh(x) + 1) / 2


def sigmoid_tanh_plain(x):
    """non-CUDA version of sigmoid_tanh"""
    return (math.tanh(x) + 1) / 2


def feedf(inputs, weights, bias):
    total = 0
    for i in range(len(inputs)):
        total = total + inputs[i] * weights[i]
    total = total + bias
    output = sigmoid_tanh_plain(total)
    return output


class Neuron:

    def __init__(self, name, weights, bias):
        self.name = name
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        return feedf(inputs, self.weights, self.bias)

    def mutate(self, power, maxMutation):
        """ mutation operator, used by genetic algorithm"""

        # mutating weights
        for i in range(len(self.weights)):
            weight_mutation_rate = random.random()**power * maxMutation
            self.weights[i] += random.uniform(-weight_mutation_rate, weight_mutation_rate)

        # mutating bias
        bias_mutation_rate = random.random()**power * maxMutation
        self.bias += random.uniform(-bias_mutation_rate, bias_mutation_rate)

        # clipping values
        if(CLIP_VALUES):
            self.weights = np.clip(self.weights, -1.0, 1.0)
            self.bias = np.clip(self.bias, -1.0, 1.0)


class NeuralNetwork:

    def __init__(self, hiddenLayers, neuronsInLayer):

        # initializing some values
        self.hidden_neurons = []
        self.hiddenLayers = hiddenLayers
        self.neuronsInLayer = neuronsInLayer

        # creating neurons on hidden layers
        for i in range(hiddenLayers):
            for j in range(neuronsInLayer):
                new_neuron = Neuron("h" + str(i) + ":" + str(j), [0, 0], 0)
                self.hidden_neurons.append(new_neuron)

        # creating output neuron
        o1_initial_weights = []
        for i in range(len(self.hidden_neurons)):
            o1_initial_weights.append(0)
        self.o1 = Neuron("o1", o1_initial_weights, 0)

    def feedforward(self, x):

        # calculating outputs of hidden layer neurons
        outputs = []
        for i in range(len(self.hidden_neurons)):
            output = self.hidden_neurons[i].feedforward(x)
            outputs.append(output)

        # calculating outputs of output neurons
        o1_out = self.o1.feedforward(outputs)
        return o1_out

    def mutate(self, power, maxMutation):

        # mutating hidden layer neurons
        for i in range(len(self.hidden_neurons)):
            self.hidden_neurons[i].mutate(power, maxMutation)

        # mutating output neuron
        self.o1.mutate(power, maxMutation)

    def get_weights(self):
        """
        returns list with weights of all hidden layer neurons
        useful for sending to CUDA device
        """
        weights = []
        for neuron in self.hidden_neurons:
            for weight in neuron.weights:
                weights.append(weight)
        return weights

    def get_biases(self):
        """returns list with weights of all hidden layer neurons"""
        biases = []
        for neuron in self.hidden_neurons:
            biases.append(neuron.bias)
        return biases

    def get_output_weights(self):
        """returns weights of output neuron"""
        return self.o1.weights

    def get_output_bias(self):
        """returns bias of output neuron"""
        return self.o1.bias


def lists_average(list1, list2):
    """takes two lists and makes new list, each value is an average of pair of values from these lists"""
    avg_list = []
    assert len(list1) == len(list2), "Lists have different length"
    for i in range(len(list1)):
        avg_list.append((list1[i] + list2[i]) / 2.0)
    return avg_list


def neuron_crossover(neuron1, neuron2):
    """Crossover operator for two neurons, used by genetic algorithm"""
    return Neuron(neuron1.name, lists_average(neuron1.weights, neuron2.weights), (neuron1.bias + neuron2.bias) / 2)


def crossover(network1, network2):
    """Crossover operator for two neural networks, used by genetic algorithm"""

    new_network = NeuralNetwork(network1.hiddenLayers, network1.neuronsInLayer)

    # crossover of hidden layer neurons
    for i in range(len(network1.hidden_neurons)):
        new_network.hidden_neurons[i] = neuron_crossover(network1.hidden_neurons[i], network2.hidden_neurons[i])

    # crossover of output neurons
    new_network.o1 = neuron_crossover(network1.o1, network2.o1)

    return new_network


@cuda.jit(device=True)
def NNfeedf(hWeights, hBiases, oWeights, oBias, x, y):
    """
    Feedforward function
    Runs on CUDA device
    """

    # CUDA array for temporarily storing outputs of hidden layer neurons
    outputs = numba.cuda.local.array(MAX_NEURONS, float32)

    # calculating outputs of hidden layer neurons
    for i in range(len(hBiases)):
        output = sigmoid_tanh(x * hWeights[i * 2] + y * hWeights[i * 2 + 1] + hBiases[i])
        outputs[i] = output

    # calculating output of the output neuron
    o1_out = 0
    for i in range(len(outputs)):
        o1_out = o1_out + outputs[i] * oWeights[i]
    o1_out = o1_out + oBias
    o1_out = sigmoid_tanh(o1_out)

    return o1_out


def NNfeedf_plain(hWeights, hBiases, oWeights, oBias, x, y):
    """Non-CUDA version of NNfeedf"""

    # calculating outputs of hidden layer neurons
    outputs = []
    for i in range(len(hBiases)):
        output = sigmoid_tanh_plain(x * hWeights[i * 2] + y * hWeights[i * 2 + 1] + hBiases[i])
        outputs.append(output)

    # calculating output of the output neuron
    o1_out = 0
    for i in range(len(outputs)):
        o1_out = o1_out + outputs[i] * oWeights[i]
    o1_out = o1_out + oBias
    o1_out = sigmoid_tanh_plain(o1_out)

    return o1_out
