import numpy as np
import pandas as pd
import random
import math
import graphics
import multiprocessing
import numba
from numba import float32
from numba import cuda
import time

# genetic algorithm settings
POPULATION_SIZE = 10            # amount of neural networks in each generation
CROSSOVER_POWER = 2             # increasing this number will cause best network to be more likey to reproduce
MUTATION_POWER = 100            # how likely small mutations are
MAX_MUTATION = 1000             # limits mutation of weights to that amount at once
ITERATIONS = 1000               # generation limit
MINIMAL_ERROR_SHUTDOWN = False  # stop if error is small enough

# neural network settings
HIDDEN_LAYER_NEURONS = 3        # number of neurons on the hidden layer
CLIP_VALUES = False             # limit weights and biases to 0.0..1.0

# output settings
PRINT_WEIGHTS = False           # print weights of all neurons every generation
PRINT_GEN_NUMBER = True         # print generation number every generation
RENDER_EVERY = 10               # render every N generation, useful if there are a lot of neurons and render is too slow

last_id = 0                     # global variable for last used id of network, used to assign ids


def increase_last_id():
    global last_id
    last_id += 1


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

    def mutate(self):
        """ mutation operator, used by genetic algorithm"""

        # mutating weights
        for i in range(len(self.weights)):
            weight_mutation_rate = random.random()**MUTATION_POWER * MAX_MUTATION
            self.weights[i] += random.uniform(-weight_mutation_rate, weight_mutation_rate)

        # mutating bias
        bias_mutation_rate = random.random()**MUTATION_POWER * MAX_MUTATION
        self.bias += random.uniform(-bias_mutation_rate, bias_mutation_rate)

        # clipping values
        if(CLIP_VALUES):
            self.weights = np.clip(self.weights, -1.0, 1.0)
            self.bias = np.clip(self.bias, -1.0, 1.0)


class NeuralNetwork:

    def __init__(self):

        # initializing some values
        self.hidden_neurons = []
        self.id = last_id
        increase_last_id()
        self.parent1 = -1
        self.parent2 = -1

        # creating neurons on hidden layer
        for i in range(HIDDEN_LAYER_NEURONS):
            new_neuron = Neuron("h" + str(i), [0, 0], 0)
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

    def mutate(self):

        # mutating hidden layer neurons
        for i in range(len(self.hidden_neurons)):
            self.hidden_neurons[i].mutate()

        # mutating output neuron
        self.o1.mutate()

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

    new_network = NeuralNetwork()

    # crossover of hidden layer neurons
    for i in range(len(network1.hidden_neurons)):
        new_network.hidden_neurons[i] = neuron_crossover(network1.hidden_neurons[i], network2.hidden_neurons[i])

    # crossover of output neurons
    new_network.o1 = neuron_crossover(network1.o1, network2.o1)

    return new_network


def center_column(data_frame, column_name):
    """centers column of numbers stored in pandas DataFrame around mean value of these numbers"""
    mean = np.mean(list(data_frame[column_name]))
    for i in range(len(data_frame.index)):
        data_frame.iloc[i, data_frame.columns.get_loc(column_name)] -= mean
    return mean


@cuda.jit(device=True)
def NNfeedf(hWeights, hBiases, oWeights, oBias, x, y):
    """
    Feedforward function
    Runs on CUDA device
    """

    # CUDA array for temporarily storing outputs of hidden layer neurons
    outputs = numba.cuda.local.array(HIDDEN_LAYER_NEURONS, float32)

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


@cuda.jit
def render_graph(hWeights, hBiases, oWeights, oBias, points):
    """Calculates outputs of best neural network for rendering them in a separate window"""

    # since window dimensions can differ from data dimensions, values have to be scaled
    scaleFactorX = graphics.SCR_WIDTH / graphics.DATA_MAX_X
    scaleFactorY = graphics.SCR_HEIGHT / graphics.DATA_MAX_Y

    # CUDA thread index
    pos = cuda.grid(1)

    # calculating x and y positions from CUDA thread index
    y = pos // graphics.ARR_SIZE_Y * graphics.STEP_Y
    x = pos % graphics.ARR_SIZE_Y * graphics.STEP_X

    # running neural network
    result = NNfeedf(hWeights, hBiases, oWeights, oBias, int(x / scaleFactorX - weight_mean + graphics.STEP_X / 2.0), int(y / scaleFactorY - height_mean + graphics.STEP_Y / 2.0))

    # putting result in the output array
    points[pos] = result


def calculate_errors_plain(weights, heights, genders, hWeights, hBiases, oWeights, oBiases):
    """Non-CUDA version of calculate_errors function"""
    network_mean_errors = []
    for i in range(POPULATION_SIZE):
        errors = []

        weights_start = i * HIDDEN_LAYER_NEURONS * 2
        weights_end = weights_start + HIDDEN_LAYER_NEURONS * 2
        current_hWeights = hWeights[weights_start:weights_end]

        biases_start = i * HIDDEN_LAYER_NEURONS
        biases_end = biases_start + HIDDEN_LAYER_NEURONS
        current_hBiases = hBiases[biases_start:biases_end]

        oweights_start = i * HIDDEN_LAYER_NEURONS
        oweights_end = oweights_start + HIDDEN_LAYER_NEURONS
        current_oWeights = oWeights[oweights_start:oweights_end]

        oBias = oBiases[i]

        for j in range(len(weights)):
            # result = generation[i].feedforward([weights[j], heights[j]])
            result = NNfeedf_plain(current_hWeights, current_hBiases, current_oWeights, oBias, weights[j], heights[j])  # change to NNfeedf
            gender = 0 if genders[j] == "M" else 1
            error = abs(result - gender)
            errors.append(error)
        mean_error = np.mean(np.array(errors))
        # print(mean_error)
        network_mean_errors.append(mean_error)
    return network_mean_errors


@cuda.jit
def calculate_errors(weights, heights, genders, hWeights, hBiases, oWeights, oBiases, errors_out):
    """
    Calculates error of a neural network on one data point
    Runs on CUDA device
    """

    # index of CUDA thread
    pos = cuda.grid(1)
    # index of neural network
    i = pos // len(weights)
    # index of data point
    j = pos % len(weights)

    # getting weights of neural network from hWeights array
    weights_start = i * HIDDEN_LAYER_NEURONS * 2
    weights_end = weights_start + HIDDEN_LAYER_NEURONS * 2
    current_hWeights = hWeights[weights_start:weights_end]

    # getting biases of neural network from hBiases array
    biases_start = i * HIDDEN_LAYER_NEURONS
    biases_end = biases_start + HIDDEN_LAYER_NEURONS
    current_hBiases = hBiases[biases_start:biases_end]

    # getting output neuron weight weights from oWeights array
    oweights_start = i * HIDDEN_LAYER_NEURONS
    oweights_end = oweights_start + HIDDEN_LAYER_NEURONS
    current_oWeights = oWeights[oweights_start:oweights_end]

    # getting output neuron bias
    oBias = oBiases[i]

    # running neural network
    result = NNfeedf(current_hWeights, current_hBiases, current_oWeights, oBias, weights[j], heights[j])

    # resulting error is difference between the output of the network and data point value
    error = abs(result - genders[j])

    # putting result in the output array
    errors_out[pos] += error


def train(weights, heights, genders):
    """Trains neural network"""

    # creating networks
    generation = []
    for i in range(POPULATION_SIZE):
        new_network = NeuralNetwork()
        generation.append(new_network)

    # minimal error starts at 1.0 at first and gets smaller later
    minimal_error = 1.0

    for iteration in range(ITERATIONS):

        # if you want to see the weights of all neurons
        if(PRINT_WEIGHTS):
            for i in range(POPULATION_SIZE):
                print("Network " + str(i) + ":    " + str(generation[i].parent1) + " " + str(generation[i].parent2))
                for j in range(len(generation[i].hidden_neurons)):
                    neuron = generation[i].hidden_neurons[j]
                    print("    " + neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))

        # calculating error

        # getting values of the neural netwok
        # this has to be done to run the network on CUDA device
        hWeights = []
        hBiases = []
        oWeights = []
        oBiases = []
        for i in range(POPULATION_SIZE):
            current_network = generation[i]
            hWeights = hWeights + current_network.get_weights()
            hBiases = hBiases + current_network.get_biases()
            oWeights = oWeights + current_network.get_output_weights()
            oBiases.append(current_network.get_output_bias())

        # errors will be put in this array
        network_errors_raw = np.zeros(POPULATION_SIZE * len(weights))

        # converting array of genders (which are represented by "M" and "F") to array of numbers
        genders_num = [0 if gender == "M" else 1 for gender in genders]

        # calculating errors of neural networks on
        calculate_errors[POPULATION_SIZE, len(weights)](np.array(weights), np.array(heights), np.array(genders_num), np.array(hWeights), np.array(hBiases), np.array(oWeights), np.array(oBiases), network_errors_raw)

        # calculate_errors makes array of individual erros on every data point
        # to calculate fitness of the neural network, we need to know its average error
        network_errors_mean = [0] * POPULATION_SIZE
        for i in range(POPULATION_SIZE):
            s = 0
            for j in range(len(weights)):
                s += network_errors_raw[i * len(weights) + j]
            network_errors_mean[i] = s
        for i in range(POPULATION_SIZE):
            network_errors_mean[i] /= len(weights)

        # calculating fitness
        for i in range(POPULATION_SIZE):
            if network_errors_mean[i] != 0:
                generation[i].fitness = 1.0 / network_errors_mean[i]
            else:
                generation[i].fitness = float("inf")

        # list has to be sorted
        generation.sort(key=lambda x: x.fitness, reverse=True)

        # updating minimal error
        if(1.0 / generation[0].fitness < minimal_error):
            minimal_error = 1.0 / generation[0].fitness

        # creating new generation
        new_generation = []
        for i in range(POPULATION_SIZE):

            # preserving the best network
            if i == 0:
                new_network = crossover(generation[0], generation[0])
                new_network.parent1 = generation[0].id
                new_network.parent2 = generation[0].id
                new_generation.append(new_network)
                continue

            # choosing parents
            rand1 = random.random()**CROSSOVER_POWER
            rand2 = random.random()**CROSSOVER_POWER
            scaledRand1 = rand1 * POPULATION_SIZE
            scaledRand2 = rand2 * POPULATION_SIZE
            pick1 = int(scaledRand1)
            pick2 = int(scaledRand2)

            # crossover and mutation
            new_network = crossover(generation[pick1], generation[pick2])
            new_network.parent1 = generation[pick1].id
            new_network.parent2 = generation[pick2].id
            new_network.mutate()
            new_generation.append(new_network)

        # swapping generations
        generation = new_generation

        # rendering results in a separate window
        if(iteration % RENDER_EVERY == 0 or minimal_error == 0.0):

            # we want to use not just any networ, but the best one
            best_network = generation[0]

            points = np.zeros(graphics.ARR_SIZE_X * graphics.ARR_SIZE_Y)

            # calculating grid of values
            render_graph[graphics.ARR_SIZE_X, graphics.ARR_SIZE_Y](np.array(best_network.get_weights()), np.array(best_network.get_biases()), np.array(best_network.get_output_weights()), best_network.get_output_bias(), points)

            # sending resulting list to the renderer
            points_queue.put(points)

        if PRINT_GEN_NUMBER:
            print("Generation " + str(iteration + 1) + " " + str(minimal_error), end="\r")

        # if minimal error goes below threshold, training stops
        if MINIMAL_ERROR_SHUTDOWN:
            if minimal_error < 1.0 / len(weights) / 2:
                break

        # if minimal error is zero, there is no point to continue training
        if minimal_error == 0.0:
            break

    print()

    # returning best network
    return generation[0]


# main function
if __name__ == '__main__':

    # setting data
    data = [

        ["Alice", 123, 65, "F"],
        ["Bob", 160, 72, "M"],
        ["Charlie", 152, 70, "M"],
        ["Diana", 120, 60, "F"],
        ["Eugene", 164, 69, "M"],
        ["Fiona", 129, 65, "F"],
        ["Garreth", 177, 75, "M"],
        ["Heather", 135, 55, "F"],

        ["Short man 1", 75, 30, "M"],
        ["Short man 2", 70, 25, "M"],
        ["Short man 3", 80, 28, "M"],
        ["Short man 4", 90, 50, "M"],
        ["Short heavy man 1", 75, 150, "M"],
        ["Short heavy man 2", 70, 125, "M"],
        ["Short heavy man 3", 80, 134, "M"],
        ["Short heavy man 4", 90, 128, "M"],
        ["Short woman 1", 49, 78, "F"],
        ["Short woman 2", 58, 74, "F"],
        ["Short woman 3", 32, 90, "F"],
        ["Short woman 4", 56, 66, "F"],
        ["Tall light man 1", 180, 23, "M"],
        ["Tall light man 2", 170, 20, "M"],
        ["Tall light man 3", 175, 30, "M"],
        ["Tall light man 4", 169, 10, "M"],

        ["1", 10, 148, "F"],
        ["1", 15, 126, "F"],
        ["1", 16, 131, "F"],
        ["1", 20, 143, "F"],
        ["1", 30, 28, "F"],
        ["1", 40, 70, "F"],
        ["1", 50, 179, "F"],
        ["1", 60, 62, "F"],
        ["1", 70, 50, "F"],
        ["1", 80, 65, "F"],
        ["1", 90, 32, "F"],
        ["2", 19, 156, "M"],
        ["2", 120, 58, "M"],
        ["2", 93, 22, "M"],
        ["2", 191, 120, "M"],
        ["2", 146, 135, "M"],

    ]

    # randomly generating data
    data = []
    for i in range(1000):
        gender = "M" if random.random() < 0.5 else "F"
        data.append([str(i), random.random()**3 * 200, random.uniform(0, 200), gender])

    # putting data in pandas DataFrame
    df = pd.DataFrame(data, columns=["Name", "Weight", "Height", "Gender"])

    # data has to be centered before processing
    weight_mean = center_column(df, "Weight")
    height_mean = center_column(df, "Height")

    # initializing renderer
    renderer = graphics.Graphics()
    points_queue = multiprocessing.Queue()

    # creating list of data points for the renderer
    data_points = []
    weight_column = list(df["Weight"])
    height_column = list(df["Height"])
    for i in range(len(weight_column)):
        point_color = (0, 0, 255) if df.loc[i]["Gender"] == "M" else (255, 0, 0)
        new_point = graphics.Point(weight_column[i] + weight_mean, height_column[i] + height_mean, point_color)
        data_points.append(new_point)

    # strting renderer
    renderer.start(points_queue, data_points)

    time1 = time.time()

    # training
    best_network = train(list(df["Weight"]), list(df["Height"]), list(df["Gender"]))

    print()

    # printing weights of the best network
    print("Weights")
    for neuron in best_network.hidden_neurons:
        print(neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))
    print(best_network.o1.name)
    for i in range(len(best_network.hidden_neurons)):
        print("    " + best_network.hidden_neurons[i].name + " " + str(best_network.o1.weights[i]))

    print()

    # resulting time
    time2 = time.time()
    print(f"Time: {time2 - time1}")
