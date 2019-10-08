import numpy as np
import random
import math
import numba
from numba import float32
from numba import cuda
import topogroup
import copy

neuron_id = 0
network_id = 0

CLIP_VALUES = False             # limit weights and biases to 0.0..1.0
MAX_NEURONS = 100               # needed for memory allocation on CUDA


def sigmoid_exp(x):
    """
    Sigmoid function, calculated using exponent.
    """
    return 1 / (1 + np.exp(-x))


@cuda.jit(device=True)
def sigmoid_tanh(x):
    """
    Same as sigmoid_exp, but calculated using tanh.
    sigmoid_exp gives overflow error if weights are too high, but this function does not.
    """
    return (math.tanh(x) + 1) / 2


def sigmoid_tanh_plain(x):
    """
    non-CUDA version of sigmoid_tanh
    """
    return (math.tanh(x) + 1) / 2


class _Neuron:

    def __init__(self, name):
        self.name = name
        self.inputLinks = []
        self.outputLinks = []
        self.function = math.tanh
        self.groupId = 0
        self.inputCount = 0
        global neuron_id
        self.id = neuron_id
        neuron_id += 1

    def addLink(self, neuron):
        """
        Connects this neuron to another neuron.
        """
        neuron.weights.append(0)
        neuron.inputCount += 1
        neuron.inputLinks.append(self)
        self.outputLinks.append(neuron)

    def __repr__(self):
        print(self.id)


class InputNeuron(_Neuron):

    def __init__(self, name):
        _Neuron.__init__(self, name)
        self.value = 0
        self.bias = 0

    def feedforward(self):
        pass

    def __repr__(self):
        return f"{self.name}->{self.value}"


class Neuron(_Neuron):

    def __init__(self, name, weights=[], bias=0):
        _Neuron.__init__(self, name)
        self.weights = weights.copy()
        self.bias = bias
        self.value = 0

    def feedforward(self):
        """
        Calculates output of neuron using inputs, weights and bias.
        """
        inputs = [neuron.value for neuron in self.inputLinks]
        total = 0
        for i in range(len(inputs)):
            total = total + inputs[i] * self.weights[i]
        total = total + self.bias
        output = self.function(total)
        self.value = output

    def mutate(self, power, maxMutation):
        """
        Mutation operator, used by genetic algorithm.
        """
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

    def __repr__(self):
        return f"{self.name}-{self.weights}({self.bias})->{self.value}"


class NeuralNetwork:

    def __init__(self, hiddenLayers, neuronsInLayer):

        self.hiddenLayers = hiddenLayers
        self.neuronsInLayer = neuronsInLayer

        # id is useful for debugging
        global network_id
        self.id = network_id
        network_id += 1

        # used by genetic algorithm
        self.error = 1

        # lists of neurons
        self.neurons = []
        self.inputNeurons = []
        self.hiddenNeurons = []

        # creating input neurons
        input1 = InputNeuron("x")
        input2 = InputNeuron("y")
        self.addInputNeuron(input1)
        self.addInputNeuron(input2)

        # creating hidden layer neurons
        previousLayer = []
        currentLayer = []
        for layer_i in range(hiddenLayers):
            currentLayer = []
            # creating new layer of neurons
            for neuron_i in range(neuronsInLayer):
                new_neuron = Neuron("h" + str(layer_i) + ":" + str(neuron_i))
                self.addHiddenNeuron(new_neuron)
                currentLayer.append(new_neuron)
            # connecting input neurons to hidden neurons
            if layer_i == 0:
                for inputNeuron in self.inputNeurons:
                    for hiddenNeuron in self.hiddenNeurons:
                        self.connect(inputNeuron, hiddenNeuron)
            # connecting neurons in current layer to neurons in previous layer
            if layer_i > 0:
                for previousLayerNeuron in previousLayer:
                    for currentLayerNeuron in currentLayer:
                        self.connect(previousLayerNeuron, currentLayerNeuron)
            previousLayer = currentLayer

        # to connect output neuron to the last layer, we need to know where that layer is
        self.sortNeurons()

        # creating output neuron
        outputNeuron = Neuron("o1")
        outputNeuron.function = sigmoid_tanh_plain
        self.addHiddenNeuron(outputNeuron)

        # connecting neurons in the last hidden layer to the output neuron
        for neuron in self.layers[-1]:
            self.connect(neuron, outputNeuron)

        # because output neuron was added, neurons have to be sorted again
        self.sortNeurons()

    def connect(self, neuron1, neuron2):
        """
        Connects two neurons.
        """
        neuron1.addLink(neuron2)

    def addHiddenNeuron(self, neuron):
        self.hiddenNeurons.append(neuron)
        self.neurons.append(neuron)

    def addInputNeuron(self, neuron):
        self.inputNeurons.append(neuron)
        self.neurons.append(neuron)

    def sortNeurons(self):
        """
        Sorts neurons in layers.
        Does not affect feedforward.
        Useful for sending neurons to GPU in layers.
        """
        self.layers = topogroup.groupNodes(self.neurons)

    def feedforward(self, inputs):
        """
        Runs neural network.
        """
        # input neurons must have value set
        self.setInputs(inputs)

        # calculating outputs of hidden layer neurons
        for layer in self.layers:
            for neuron in layer:
                neuron.feedforward()

        # returning values of neurons on the last layer
        return [neuron.value for neuron in self.layers[-1]]

    def mutate(self, power, maxMutation):
        """
        Mutates hidden layer neurons
        """
        for neuron in self.hiddenNeurons:
            neuron.mutate(power, maxMutation)

    def copy(self):
        """
        Creates deep copy of the network.
        """
        return copy.deepcopy(self)

    def setInputs(self, inputs):
        """
        Sets values to input neurons.
        """
        for i in range(len(inputs)):
            self.inputNeurons[i].value = inputs[i]

    def getWeightsMatrix(self):
        matrix_size = len(self.neurons)
        matrix = np.zeros((matrix_size, matrix_size))
        for left in range(matrix_size):
            for right in range(matrix_size):
                # if self.neurons[right] in self.neurons[left].outputLinks:
                #     index_in_weights = self.neurons[right].inputLinks.index(self.neurons[left])
                #     matrix[left, right] = self.neurons[right].weights[index_in_weights]
                if self.neurons[left] in self.neurons[right].inputLinks:
                    index_in_weights = self.neurons[right].inputLinks.index(self.neurons[left])
                    matrix[right, left] = self.neurons[right].weights[index_in_weights]
        return matrix

    def getBiases(self):
        return [neuron.bias for neuron in self.neurons]

    def getValues(self):
        return [neuron.value for neuron in self.neurons]

    def __repr__(self):
        s = f"id: {self.id}\n"
        for neuron in self.neurons:
            s += f"    {neuron.__repr__()}\n"
        return s


def lists_average(list1, list2):
    """
    Takes two lists and makes new list, each value is an average of pair of values from these lists
    """
    avg_list = []
    assert len(list1) == len(list2), "Lists have different length"
    for i in range(len(list1)):
        avg_list.append((list1[i] + list2[i]) / 2.0)
    return avg_list


@cuda.jit(device=True)
def NNfeedf(weightsMatrix, biases, neuronCount, x, y):
    """
    Feedforward function
    Runs on CUDA device
    """

    # CUDA array for storing outputs
    values = numba.cuda.local.array(MAX_NEURONS, float32)
    values[0] = x
    values[1] = y
    for neuron_i in range(neuronCount):
        for other_neuron_i in range(neuronCount):
            values[neuron_i] += values[other_neuron_i] * weightsMatrix[neuron_i, other_neuron_i]
        values[neuron_i] += biases[neuron_i]
        if neuron_i >= 2:
            values[neuron_i] = math.tanh(values[neuron_i])

    return (values[neuronCount - 1] + 1) / 2.0


def NNfeedf_plain(weightsMatrix, biases, neuronCount, x, y):
    """
    Non-CUDA version of NNfeedf
    """
    values = np.zeros(MAX_NEURONS)
    values[0] = x
    values[1] = y
    for neuron_i in range(neuronCount):
        for other_neuron_i in range(neuronCount):
            values[neuron_i] += values[other_neuron_i] * weightsMatrix[neuron_i, other_neuron_i]
        values[neuron_i] += biases[neuron_i]
        if neuron_i >= 2:
            values[neuron_i] = math.tanh(values[neuron_i])
    result = (values[neuronCount - 1] + 1) / 2.0
    return result


@cuda.jit
def process_array(array, N):
    pos = cuda.grid(1)
    row = pos // N
    column = pos % N
    array[row, column] *= 2.0


# @cuda.jit
# def cuda_layered_render(weightsMatrix, biases, values, isFirstHiddenLayer):
#     # print(f"Weights matrix:\n{weightsMatrix}")
#     # print(f"Biases:\n{biases}")
#     # print(f"Values:\n{values}")
#     # print(f"IsFirst: {isFirstHiddenLayer}")
#     # print()
#     pos = cuda.grid(1)
#     x = pos // graphics.ARR_SIZE_X * graphics.STEP_X
#     y = pos % graphics.ARR_SIZE_X * graphics.STEP_Y


# def layered_render(network):
#     total_neuron_count = len(network.neurons)
#     weights_matrix = network.getWeightsMatrix()
#     biases = np.array(network.getBiases())
#     values_size = graphics.ARR_SIZE_X * graphics.ARR_SIZE_Y * total_neuron_count
#     values = np.zeros(values_size)
#     for layer_i in range(1, len(network.layers)):
#         cuda_layered_render(weights_matrix, biases, values, layer_i == 1)


if __name__ == "__main__":

    import graphics
    import multiprocessing

    network = NeuralNetwork(1, 2)

    network.layers[1][0].weights[0] = 1
    network.layers[1][1].weights[0] = 1
    network.layers[2][0].weights[0] = 1
    # network.layers[3][0].weights[0] = 1
    # network.layers[4][0].weights[0] = 1
    # network.layers[5][0].weights[0] = 1
    # # network.layers[1][0].weights[1] = -1
    # # network.layers[2][0].weights[0] = 1
    # network.layers[-1][0].weights[0] = 1

    # result = network.feedforward([2, 3])
    print(network)

    matrix = network.getWeightsMatrix()
    print(matrix)

    # layered_render(network)
    N = len(network.neurons)
    process_array[N, N](matrix, len(network.neurons))
    print(matrix)

    # weight_mean = 141.5
    # height_mean = 68.5

    # data_points = []
    # data_points.append(graphics.Point(123, 65, (255, 0, 0)))
    # data_points.append(graphics.Point(160, 72, (0, 0, 255)))

    # renderer = graphics.Graphics()
    # points_queue = multiprocessing.Queue()
    # renderer.start(points_queue, data_points)

    # points = []
    # scaleFactorX = graphics.SCR_WIDTH / graphics.DATA_MAX_X
    # scaleFactorY = graphics.SCR_HEIGHT / graphics.DATA_MAX_Y
    # for y in range(graphics.ARR_SIZE_Y):
    #     for x in range(graphics.ARR_SIZE_X):
    #         result = network.feedforward([int(x * graphics.STEP_X / scaleFactorX - weight_mean + graphics.STEP_X / 2.0), int(y * graphics.STEP_Y / scaleFactorY - height_mean + graphics.STEP_Y / 2.0)])[0]
    #         points.append(result)

    # points_queue.put(points)
