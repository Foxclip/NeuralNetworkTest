import numpy as np
import pandas as pd
import random
import graphics
import multiprocessing
from numba import cuda
import time
import network
import math
import sys

# genetic algorithm settings
POPULATION_SIZE = 2             # amount of neural networks in each generation
CROSSOVER_POWER = 2             # increasing this number will cause best network to be more likey to reproduce
MUTATION_POWER = 100            # how likely small mutations are
MAX_MUTATION = 0.1              # limits mutation of weights to that amount at once
ITERATIONS = 10000              # generation limit
MINIMAL_ERROR_SHUTDOWN = False  # stop if error is small enough

# neural network settings
HIDDEN_LAYER_NEURONS = 1        # number of neurons in the hidden layer
HIDDEN_LAYERS = 1               # number of hidden layers

# output settings
PRINT_WEIGHTS = False           # print weights of all neurons every generation
PRINT_GEN_NUMBER = True         # print generation number every generation
RENDER_EVERY = 10               # render every N generation, useful if there are a lot of neurons and render is too slow

last_id = 0                     # global variable for last used id of network, used to assign ids


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
    result = network.NNfeedf(hWeights, hBiases, oWeights, oBias, int(x / scaleFactorX - weight_mean + graphics.STEP_X / 2.0), int(y / scaleFactorY - height_mean + graphics.STEP_Y / 2.0))

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
            result = network.NNfeedf_plain(current_hWeights, current_hBiases, current_oWeights, oBias, weights[j], heights[j])  # change to NNfeedf
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
    result = network.NNfeedf(current_hWeights, current_hBiases, current_oWeights, oBias, weights[j], heights[j])

    # resulting error is difference between the output of the network and data point value
    error = abs(result - genders[j])

    # putting result in the output array
    errors_out[pos] += error


def center_column(data_frame, column_name):
    """centers column of numbers stored in pandas DataFrame around mean value of these numbers"""
    mean = np.mean(list(data_frame[column_name]))
    for i in range(len(data_frame.index)):
        data_frame.iloc[i, data_frame.columns.get_loc(column_name)] -= mean
    return mean


def train(weights, heights, genders):
    """Trains neural network"""

    # creating networks
    generation = []
    for i in range(POPULATION_SIZE):
        new_network = network.NeuralNetwork(HIDDEN_LAYERS, HIDDEN_LAYER_NEURONS)
        generation.append(new_network)

    # minimal error starts at 1.0 at first and gets smaller later
    minimal_error = 1.0

    for iteration in range(ITERATIONS):

        # if you want to see the weights of all neurons
        if(PRINT_WEIGHTS):
            for i in range(POPULATION_SIZE):
                print("Network " + str(i))
                for j in range(len(generation[i].hidden_neurons)):
                    neuron = generation[i].hidden_neurons[j]
                    print("    " + neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))

        # calculating error

        # # getting values of the neural netwok
        # # this has to be done to run the network on CUDA device
        # hWeights = []
        # hBiases = []
        # oWeights = []
        # oBiases = []
        # for i in range(POPULATION_SIZE):
        #     current_network = generation[i]
        #     hWeights = hWeights + current_network.get_weights()
        #     hBiases = hBiases + current_network.get_biases()
        #     oWeights = oWeights + current_network.get_output_weights()
        #     oBiases.append(current_network.get_output_bias())

        # # errors will be put in this array
        # network_errors_raw = np.zeros(POPULATION_SIZE * len(weights))

        # # converting array of genders (which are represented by "M" and "F") to array of numbers
        # genders_num = [0 if gender == "M" else 1 for gender in genders]

        # # calculating errors of neural networks on
        # calculate_errors[POPULATION_SIZE, len(weights)](np.array(weights), np.array(heights), np.array(genders_num), np.array(hWeights), np.array(hBiases), np.array(oWeights), np.array(oBiases), network_errors_raw)

        # # calculate_errors makes array of individual erros on every data point
        # # to calculate fitness of the neural network, we need to know its average error
        # network_errors_mean = [0] * POPULATION_SIZE
        # for i in range(POPULATION_SIZE):
        #     s = 0
        #     for j in range(len(weights)):
        #         s += network_errors_raw[i * len(weights) + j]
        #     network_errors_mean[i] = s
        # for i in range(POPULATION_SIZE):
        #     network_errors_mean[i] /= len(weights)

        # for neuron in generation[0].hiddenNeurons:
        #     print(f"{neuron.name} {neuron.weights}")

        # print(generation[1])
        # for neuron in generation[1].hiddenNeurons:
        #     print(f"{neuron.name} {neuron.weights}")

        network_errors_mean = [0] * POPULATION_SIZE
        for i in range(len(generation)):
            currentNetwork = generation[i]
            # print(currentNetwork)
            # for neuron in currentNetwork.hiddenNeurons:
            #     print(f"{neuron.name} {neuron.weights}")
            for j in range(len(weights)):
                result = currentNetwork.feedforward([weights[i], heights[i]])[0]
                error = abs(result - (0 if genders[i] == "M" else 1))
                network_errors_mean[i] += error
            network_errors_mean[i] /= len(weights)
        # print()

        # calculating fitness
        for i in range(POPULATION_SIZE):
            generation[i].fitness = 1.0 / (network_errors_mean[i] + 1.0)

        # list has to be sorted
        generation.sort(key=lambda x: x.fitness, reverse=True)

        # updating minimal error
        if(1.0 / generation[0].fitness - 1 < minimal_error):
            minimal_error = 1.0 / generation[0].fitness - 1

        # creating new generation
        new_generation = []
        for i in range(POPULATION_SIZE):

            # preserving the best network
            if i == 0:
                new_network = network.crossover(generation[0], generation[0])
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
            new_network = network.crossover(generation[pick1], generation[pick2])
            new_network.mutate(MUTATION_POWER, MAX_MUTATION)
            new_generation.append(new_network)

            # print(new_network)
            # for neuron in new_network.hiddenNeurons:
            #     print(f"{neuron.name} {neuron.weights}")

            # for neuron in new_generation[0].hiddenNeurons:
            #     print(f"{neuron.name} {neuron.weights}")
            # for neuron in new_generation[1].hiddenNeurons:
            #     print(f"{neuron.name} {neuron.weights}")

        # for neuron in new_generation[0].hiddenNeurons:
        #     print(f"{neuron.name} {neuron.weights}")

        # swapping generations
        generation = new_generation

        # rendering results in a separate window
        if(iteration % RENDER_EVERY == 0 or minimal_error == 0.0):

            # we want to use not just any network, but the best one
            best_network = generation[0]

            points = []

            # points = np.zeros(graphics.ARR_SIZE_X * graphics.ARR_SIZE_Y)

            # # calculating grid of values
            # render_graph[graphics.ARR_SIZE_X, graphics.ARR_SIZE_Y](np.array(best_network.get_weights()), np.array(best_network.get_biases()), np.array(best_network.get_output_weights()), best_network.get_output_bias(), points)

            for i in range(graphics.ARR_SIZE_X):
                for j in range(graphics.ARR_SIZE_Y):
                    result = best_network.feedforward([i * graphics.STEP_X, j * graphics.STEP_Y])[0]
                    points.append(result)

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

        # print()
        # if(iteration >= 1):
        #     sys.exit()

    print()

    # returning best network
    return generation[0]


# main function
if __name__ == '__main__':

    # setting data
    data = [

        ["Alice", 123, 65, "F"],
        ["Bob", 160, 72, "M"],
        # ["Charlie", 152, 70, "M"],
        # ["Diana", 120, 60, "F"],
        # ["Eugene", 164, 69, "M"],
        # ["Fiona", 129, 65, "F"],
        # ["Garreth", 177, 75, "M"],
        # ["Heather", 135, 55, "F"],

        # ["Short man 1", 75, 30, "M"],
        # ["Short man 2", 70, 25, "M"],
        # ["Short man 3", 80, 28, "M"],
        # ["Short man 4", 90, 50, "M"],
        # ["Short heavy man 1", 75, 150, "M"],
        # ["Short heavy man 2", 70, 125, "M"],
        # ["Short heavy man 3", 80, 134, "M"],
        # ["Short heavy man 4", 90, 128, "M"],
        # ["Short woman 1", 49, 78, "F"],
        # ["Short woman 2", 58, 74, "F"],
        # ["Short woman 3", 32, 90, "F"],
        # ["Short woman 4", 56, 66, "F"],
        # ["Tall light man 1", 180, 23, "M"],
        # ["Tall light man 2", 170, 20, "M"],
        # ["Tall light man 3", 175, 30, "M"],
        # ["Tall light man 4", 169, 10, "M"],

        # ["1", 10, 148, "F"],
        # ["1", 15, 126, "F"],
        # ["1", 16, 131, "F"],
        # ["1", 20, 143, "F"],
        # ["1", 30, 28, "F"],
        # ["1", 40, 70, "F"],
        # ["1", 50, 179, "F"],
        # ["1", 60, 62, "F"],
        # ["1", 70, 50, "F"],
        # ["1", 80, 65, "F"],
        # ["1", 90, 32, "F"],
        # ["2", 19, 156, "M"],
        # ["2", 120, 58, "M"],
        # ["2", 93, 22, "M"],
        # ["2", 191, 120, "M"],
        # ["2", 146, 135, "M"],

    ]

    # # randomly generating data
    # data = []
    # # for i in range(1000):
    # #     gender = "M" if random.random() < 0.5 else "F"
    # #     data.append([str(i), random.random()**3 * 200, random.uniform(0, 200), gender])

    # for i in range(500):
    #     angle = random.random() * 360
    #     radius = random.uniform(50, 100)
    #     x = math.cos(angle * math.pi / 180) * radius + 100
    #     y = math.sin(angle * math.pi / 180) * radius + 100
    #     data.append([str(i), x, y, "F"])
    # for i in range(200):
    #     angle = random.random() * 360
    #     radius = random.uniform(10, 50)
    #     x = math.cos(angle * math.pi / 180) * radius + 100
    #     y = math.sin(angle * math.pi / 180) * radius + 100
    #     data.append([str(i), x, y, "M"])

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
    for neuron in best_network.hiddenNeurons:
        print(neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))

    print()

    # resulting time
    time2 = time.time()
    print(f"Time: {time2 - time1}")
