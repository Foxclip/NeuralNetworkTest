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

POPULATION_SIZE = 10
CROSSOVER_POWER = 2
MUTATION_POWER = 100
MAX_MUTATION = 1000
ITERATIONS = 1000
MINIMAL_ERROR_SHUTDOWN = False

HIDDEN_LAYER_NEURONS = 8
CLIP_VALUES = False

PRINT_WEIGHTS = False
RENDER_INTERPOLATION_STEP = 5
RENDER_EVERY = 10

PARALLEL_CALC_ERR = False
PARALLEL_RENDER_GRAPH = False
PARALLEL_NNFEEDF = False
PARALLEL_SIGMOID = False

last_id = 0


def increase_last_id():
    global last_id
    last_id += 1


def sigmoid_exp(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


# @njit(parallel=PARALLEL_SIGMOID)
@cuda.jit(device=True)
def sigmoid_tanh(x):
    return (math.tanh(x) + 1) / 2


def sigmoid_tanh_plain(x):
    return (math.tanh(x) + 1) / 2


# @njit
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
        # total = np.dot(self.weights, inputs) + self.bias
        # total = 0
        # for i in range(len(inputs)):
        #     total = total + inputs[i] * self.weights[i]
        # total = total + self.bias
        # output = sigmoid_tanh(total)
        # return output
        return feedf(inputs, self.weights, self.bias)

    def mutate(self):
        for i in range(len(self.weights)):
            weight_mutation_rate = random.random()**MUTATION_POWER * MAX_MUTATION
            self.weights[i] += random.uniform(-weight_mutation_rate, weight_mutation_rate)
        bias_mutation_rate = random.random()**MUTATION_POWER * MAX_MUTATION
        self.bias += random.uniform(-bias_mutation_rate, bias_mutation_rate)
        if(CLIP_VALUES):
            self.weights = np.clip(self.weights, -1.0, 1.0)
            self.bias = np.clip(self.bias, -1.0, 1.0)


class NeuralNetwork:

    def __init__(self):
        self.hidden_neurons = []
        self.id = last_id
        increase_last_id()
        self.parent1 = -1
        self.parent2 = -1
        for i in range(HIDDEN_LAYER_NEURONS):
            new_neuron = Neuron("h" + str(i), [np.random.normal(), np.random.normal()], np.random.normal())
            self.hidden_neurons.append(new_neuron)
        o1_initial_weights = []
        for i in range(len(self.hidden_neurons)):
            o1_initial_weights.append(np.random.normal())
        self.o1 = Neuron("o1", o1_initial_weights, np.random.normal())

    def feedforward(self, x):
        # for i in range(len(x)):
        #   x[i] = 1.0/x[i]
        outputs = []
        for i in range(len(self.hidden_neurons)):
            output = self.hidden_neurons[i].feedforward(x)
            outputs.append(output)
        o1_out = self.o1.feedforward(outputs)
        return o1_out

    def mutate(self):
        for i in range(len(self.hidden_neurons)):
            self.hidden_neurons[i].mutate()
        self.o1.mutate()

    def get_weights(self):
        weights = []
        for neuron in self.hidden_neurons:
            for weight in neuron.weights:
                weights.append(weight)
        return weights

    def get_biases(self):
        biases = []
        for neuron in self.hidden_neurons:
            biases.append(neuron.bias)
        return biases

    def get_output_weights(self):
        return self.o1.weights

    def get_output_bias(self):
        return self.o1.bias


def lists_average(list1, list2):
    avg_list = []
    assert len(list1) == len(list2), "Lists have different length"
    for i in range(len(list1)):
        avg_list.append(np.mean([list1[i], list2[i]]))
    return avg_list


def neuron_crossover(neuron1, neuron2):
    return Neuron(neuron1.name, lists_average(neuron1.weights, neuron2.weights), np.mean([neuron1.bias, neuron2.bias]))


def crossover(network1, network2):
    new_network = NeuralNetwork()
    for i in range(len(network1.hidden_neurons)):
        new_network.hidden_neurons[i] = neuron_crossover(network1.hidden_neurons[i], network2.hidden_neurons[i])
    new_network.o1 = neuron_crossover(network1.o1, network2.o1)
    return new_network


def center_column(data_frame, column_name):
    mean = np.mean(list(data_frame[column_name]))
    for i in range(len(data_frame.index)):
        data_frame.iloc[i, data_frame.columns.get_loc(column_name)] -= mean
    return mean


# @njit(parallel=PARALLEL_NNFEEDF)
# @njit(float32(float32[:], float32[:], float32[:], float32, float32, float32))
@cuda.jit(device=True)
def NNfeedf(hWeights, hBiases, oWeights, oBias, x, y):
    outputs = numba.cuda.local.array(128, float32)
    for i in range(len(hBiases)):
        output = sigmoid_tanh(x * hWeights[i * 2] + y * hWeights[i * 2 + 1] + hBiases[i])
        outputs[i] = output
    o1_out = 0
    for i in range(len(outputs)):
        o1_out = o1_out + outputs[i] * oWeights[i]
    o1_out = o1_out + oBias
    o1_out = sigmoid_tanh(o1_out)
    return o1_out


def NNfeedf_plain(hWeights, hBiases, oWeights, oBias, x, y):
    outputs = []
    for i in range(len(hBiases)):
        output = sigmoid_tanh_plain(x * hWeights[i * 2] + y * hWeights[i * 2 + 1] + hBiases[i])
        outputs.append(output)
    o1_out = 0
    for i in range(len(outputs)):
        o1_out = o1_out + outputs[i] * oWeights[i]
    o1_out = o1_out + oBias
    o1_out = sigmoid_tanh_plain(o1_out)
    return o1_out


# @njit(parallel=PARALLEL_RENDER_GRAPH)
@cuda.jit
def render_graph(hWeights, hBiases, oWeights, oBias, points):
    # rendering graph
    scaleFactorX = graphics.SCR_WIDTH / graphics.DATA_MAX_X
    scaleFactorY = graphics.SCR_HEIGHT / graphics.DATA_MAX_Y
    pos = cuda.grid(1)
    y = pos // graphics.ARR_SIZE_Y * graphics.STEP_Y
    x = pos % graphics.ARR_SIZE_Y * graphics.STEP_X
    result = NNfeedf(hWeights, hBiases, oWeights, oBias, int(x / scaleFactorX - weight_mean + graphics.STEP_X / 2.0), int(y / scaleFactorY - height_mean + graphics.STEP_Y / 2.0))
    points[pos] = result


# @njit(parallel=PARALLEL_CALC_ERR)
# @cuda.jit
def calculate_errors(weights, heights, genders, hWeights, hBiases, oWeights, oBiases):
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


def train(weights, heights, genders):

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
        network_mean_errors = calculate_errors(weights, heights, genders, hWeights, hBiases, oWeights, oBiases)
        # print(network_mean_errors)

        # calculating fitness
        for i in range(POPULATION_SIZE):
            if network_mean_errors[i] != 0:
                generation[i].fitness = 1.0 / network_mean_errors[i]
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

        if(iteration % RENDER_EVERY == 0 or minimal_error == 0.0):
            best_network = generation[0]
            points = np.zeros(graphics.ARR_SIZE_X * graphics.ARR_SIZE_Y)
            render_graph[graphics.ARR_SIZE_X, graphics.ARR_SIZE_Y](np.array(best_network.get_weights()), np.array(best_network.get_biases()), np.array(best_network.get_output_weights()), best_network.get_output_bias(), points)
            points_queue.put(points)

        print("Generation " + str(iteration + 1) + " " + str(minimal_error), end="\r")
        # print("Generation " + str(iteration + 1) + " " + str(minimal_error))

        # if minimal error goes below threshold, training stops
        if MINIMAL_ERROR_SHUTDOWN:
            if minimal_error < 1.0 / len(weights) / 2:
                break

        # if minimal error is zero, there is no point to continue training
        if minimal_error == 0.0:
            break

    print()

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

    df = pd.DataFrame(data, columns=["Name", "Weight", "Height", "Gender"])
    weight_mean = center_column(df, "Weight")
    height_mean = center_column(df, "Height")

    renderer = graphics.Graphics()
    # renderer.initGLFW()
    points_queue = multiprocessing.Queue()

    data_points = []
    weight_column = list(df["Weight"])
    height_column = list(df["Height"])
    for i in range(len(weight_column)):
        point_color = (0, 0, 255) if df.loc[i]["Gender"] == "M" else (255, 0, 0)
        new_point = graphics.Point(weight_column[i] + weight_mean, height_column[i] + height_mean, point_color)
        data_points.append(new_point)
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

    # testing on original data
    print()
    print("Original data")
    for j in range(len(df.index)):
        result = best_network.feedforward([df.loc[j]["Weight"], df.loc[j]["Height"]])
        result_gender = "M" if result < 0.5 else "F"
        pass_fail_string = "pass" if result_gender == df.loc[j]["Gender"] else "FAIL"
        print(df.loc[j]["Name"] + ": " + f"{result:.3f}" + " (" + str(result) + ")" + " " + pass_fail_string)

    print()

    time2 = time.time()
    print(f"Time: {time2 - time1}")
