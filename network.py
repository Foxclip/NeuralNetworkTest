import numpy as np
import random
import math
import numba
from numba import float32
from numba import cuda
import topogroup

neuron_id = 0
network_id = 0

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
        # print(f"Feedforward before {self.name} {self.value}")
        inputs = [neuron.value for neuron in self.inputLinks]
        total = 0
        for i in range(len(inputs)):
            total = total + inputs[i] * self.weights[i]
        total = total + self.bias
        output = self.function(total)
        self.value = output
        # print("    NEURON_FEED")
        # print(f"    {self.id} {self.weights}")

        # print(f"Feedforward after {self.name} {self.value} {self.weights}")

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

        # print(f"{self.name}: {self.weights}")

    def __repr__(self):
        return f"{self.name}-{self.weights}({self.bias})->{self.value}"


class NeuralNetwork:

    def __init__(self, hiddenLayers, neuronsInLayer):

        global network_id
        self.id = network_id
        network_id += 1

        self.fitness = 0

        self.hiddenLayers = hiddenLayers
        self.neuronsInLayer = neuronsInLayer

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

        self.sortNeurons()

        # creating output neuron
        outputNeuron = Neuron("o1")
        outputNeuron.function = sigmoid_tanh_plain
        self.addHiddenNeuron(outputNeuron)

        # connecting neurons in the last hidden layer to the output neuron
        for neuron in self.layers[-1]:
            self.connect(neuron, outputNeuron)

        # sorting neurons
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
        self.layers = topogroup.groupNodes(self.neurons)

    def feedforward(self, inputs):

        # print("FEED")
        # print(self)
        # for neuron in self.hiddenNeurons:
        #     print(f"{neuron.id}: {neuron.weights}")
        # print("/FEED")

        self.setInputs(inputs)

        # for layer in self.layers:
        #     for neuron in layer:
        #         try:
        #             print(f"{neuron.id}: {neuron.weights}")
        #         except Exception as e:
        #             pass

        # calculating outputs of hidden layer neurons
        for layer in self.layers:
            for neuron in layer:
                neuron.feedforward()

        # for neuron in self.neurons:
        #     print(f"{neuron.name}: {neuron.value}")

        # returning list of values of output neurons
        # print([neuron.value for neuron in self.layers[-1]])
        # sys.exit()
        return [neuron.value for neuron in self.layers[-1]]

    def mutate(self, power, maxMutation):
        # mutating hidden layer neurons
        # for neuron in self.hiddenNeurons:
        #     print(f"Before {neuron.name} {neuron.weights}")
        for neuron in self.hiddenNeurons:
            neuron.mutate(power, maxMutation)

    def setInputs(self, inputs):
        for i in range(len(inputs)):
            self.inputNeurons[i].value = inputs[i]
        # print(f"Input: {inputs[i]} Neuron name: {self.inputNeurons[i].name} Neuron value: {self.inputNeurons[i].value}")

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


def neuron_crossover(neuron, parent1, parent2):
    """
    Crossover operator for two neurons. Used by genetic algorithm.
    """
    averagedWeights = lists_average(parent1.weights, parent2.weights)
    averagedBias = (parent1.bias + parent2.bias) / 2.0
    neuron.weights = averagedWeights
    neuron.bias = averagedBias


def crossover(network1, network2):
    """
    Crossover operator for two neural networks. Used by genetic algorithm.
    """

    new_network = NeuralNetwork(network1.hiddenLayers, network1.neuronsInLayer)

    # crossover of hidden layer neurons
    for i in range(len(network1.hiddenNeurons)):
        neuron_crossover(new_network.hiddenNeurons[i], network1.hiddenNeurons[i], network2.hiddenNeurons[i])

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


if __name__ == "__main__":

    import graphics
    import multiprocessing

    network = NeuralNetwork(5, 5)

    network.layers[1][0].weights[0] = 1
    network.layers[2][0].weights[0] = 1
    network.layers[3][0].weights[0] = 1
    network.layers[4][0].weights[0] = 1
    network.layers[5][0].weights[0] = 1
    # network.layers[1][0].weights[1] = -1
    # network.layers[2][0].weights[0] = 1
    network.layers[-1][0].weights[0] = 1

    result = network.feedforward([2, 3])
    print(network)

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
