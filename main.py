import numpy as np
import pandas as pd
import random
import graphics
import multiprocessing
from numba import cuda
import time
import network
import plot

# genetic algorithm settings
POPULATION_SIZE = 10            # amount of neural networks in each generation
CROSSOVER_POWER = 2             # increasing this number will cause best network to be more likey to reproduce
MUTATION_POWER = 10             # how likely small mutations are
MAX_MUTATION = 0.1              # limits mutation of weights to that amount at once
ITERATIONS = 10000              # generation limit
MINIMAL_ERROR_SHUTDOWN = True   # stop if error is small enough

# neural network settings
HIDDEN_LAYER_NEURONS = 8        # number of neurons in the hidden layer
HIDDEN_LAYERS = 1               # number of hidden layers

# output settings
PRINT_GEN_NUMBER = True         # print generation number every generation
RENDER_EVERY = 10               # render every N generation, useful if there are a lot of neurons and render is too slow


@cuda.jit
def render_graph(weightMatrix, biases, neuronCount, points):
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
    result = network.NNfeedf(weightMatrix, biases, neuronCount, int(x / scaleFactorX - weight_mean + graphics.STEP_X / 2.0), int(y / scaleFactorY - height_mean + graphics.STEP_Y / 2.0))

    # putting result in the output array
    points[pos] = result


def center_column(data_frame, column_name):
    """centers column of numbers stored in pandas DataFrame around mean value of these numbers"""
    mean = np.mean(list(data_frame[column_name]))
    for i in range(len(data_frame.index)):
        data_frame.iloc[i, data_frame.columns.get_loc(column_name)] -= mean
    return mean


def calculate_errors(weights, heights, genders, generation):
    network_errors_mean = [0] * POPULATION_SIZE
    for i in range(len(generation)):
        currentNetwork = generation[i]
        for j in range(len(weights)):
            result = currentNetwork.feedforward([weights[j], heights[j]])[0]
            error = abs(result - (0 if genders[j] == "M" else 1))
            network_errors_mean[i] += error
        network_errors_mean[i] /= len(weights)
    return network_errors_mean


def render(best_network):

    # initializing array of points
    points = np.zeros(graphics.ARR_SIZE_X * graphics.ARR_SIZE_Y)

    # calculating grid of values
    render_graph[graphics.ARR_SIZE_X, graphics.ARR_SIZE_Y](best_network.getWeightsMatrix(), np.array(best_network.getBiases()), len(best_network.neurons), points)

    # sending resulting list to the renderer
    points_queue.put(points)


def output(iteration, minimal_error, generation):

    # rendering results in a separate window
    if(iteration % RENDER_EVERY == 0 or minimal_error == 0.0 or iteration == ITERATIONS - 1):
        render(generation[0])

    # printing gen number to console
    if PRINT_GEN_NUMBER:
        print("Generation " + str(iteration + 1) + " " + str(minimal_error), end="\r")

    # adding point to the plot
    plot_queue.put(minimal_error)


def check_stop_conditions(minimal_error, weights):

    # if minimal error goes below threshold, training stops
    if MINIMAL_ERROR_SHUTDOWN:
        if minimal_error < 1.0 / len(weights) / 2:
            return True

    # if minimal error is zero, there is no point to continue training
    if minimal_error == 0.0:
        return True


def create_generation(generation):

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

    return new_generation


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

        # calculating errors
        network_errors_mean = calculate_errors(weights, heights, genders, generation)

        # sorting list
        for i in range(POPULATION_SIZE):
            generation[i].error = network_errors_mean[i]
        generation.sort(key=lambda x: x.error)

        # updating minimal error
        if(generation[0].error < minimal_error):
            minimal_error = generation[0].error

        # creating new generation
        generation = create_generation(generation)

        # outputting results
        output(iteration, minimal_error, generation)

        if check_stop_conditions(minimal_error, weights):
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

    ]

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

    # starting plot process
    plot_queue = multiprocessing.Queue()
    plot.start(plot_queue)

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
