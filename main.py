import numpy as np
import pandas as pd
import graphics
import multiprocessing
from numba import cuda
import time
import random
import math
import network
import plot

# genetic algorithm settings
POPULATION_SIZE = 10            # amount of neural networks in each generation
MAX_MUTATION = 1                # limits mutation of weights to that amount at once
ITERATIONS = 100000             # generation limit
MINIMAL_ERROR_SHUTDOWN = False  # stop if error is small enough

# neural network settings
HIDDEN_LAYER_NEURONS = 3        # number of neurons in the hidden layer
HIDDEN_LAYERS = 1               # number of hidden layers
LEARNING_RATE = 0.001           # backpropagation learning rate

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


def calculate_sample_error(sample_index, weights, heights, genders, network):
    """
    Calculates error on one sample.
    """
    result = network.feedforward([weights[sample_index], heights[sample_index]])[0]
    error = result - (0 if genders[sample_index] == "M" else 1)
    return error * error / 2  # squared error, division is needed so after differentiation it cancels out


def calculate_errors(weights, heights, genders, generation):
    """
    Calculates errors on all samples.
    """
    network_errors_mean = [0] * POPULATION_SIZE
    for i in range(len(generation)):
        currentNetwork = generation[i]
        for j in range(len(weights)):
            error = calculate_sample_error(j, weights, heights, genders, currentNetwork)
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


def output(iteration, minimal_error, network):

    # rendering results in a separate window
    if(iteration % RENDER_EVERY == 0 or minimal_error == 0.0 or iteration == ITERATIONS - 1):
        render(network)

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


def create_generation(best_network):

    new_generation = []

    for i in range(POPULATION_SIZE):
        new_network = best_network.copy()
        if i > 0:
            power = i - POPULATION_SIZE + 1
            new_network.mutate(1, MAX_MUTATION * pow(10, power))
        new_generation.append(new_network)

    return new_generation


def train_random(weights, heights, genders):
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
        generation = create_generation(generation[0])

        # outputting results
        output(iteration, minimal_error, generation[0])

        if check_stop_conditions(minimal_error, weights):
            break

    print()

    # returning best network
    return generation[0]


def train_backprop(weights, heights, genders):
    """Trains neural network"""

    # creating network
    currentNetwork = network.NeuralNetwork(HIDDEN_LAYERS, HIDDEN_LAYER_NEURONS)

    # minimal error starts at 1.0 at first and gets smaller later
    minimal_error = 1.0

    for iteration in range(ITERATIONS):

        errors = []

        for sample_i in range(len(weights)):

            # choosing sample
            # sample_index = random.randint(0, len(weights) - 1)

            # calculating error
            network_error = calculate_sample_error(sample_i, weights, heights, genders, currentNetwork)
            errors.append(network_error)

            output_neuron = currentNetwork.layers[2][0]
            target = (0 if genders[sample_i] == "M" else 1)
            d_E_outO = output_neuron.value - target
            d_outO_netO = output_neuron.derivative(output_neuron.net)
            d_E_netO = d_E_outO * d_outO_netO
            # print(f"net: {net}")
            # print(f"outp: {outp}")
            # print(f"target: {target}")
            # print(f"d_E_o1: {d_E_o1}")
            # print(f"d_o1_n1: {d_o1_n1}")
            # print(f"weights[0]: {o1_neuron.weights[0]}")
            # print()

            for hidden_i in range(len(currentNetwork.layers[1])):
                hidden_neuron = currentNetwork.layers[1][hidden_i]
                d_netO_outH = output_neuron.weights[hidden_i]
                d_outH_netH = hidden_neuron.derivative(hidden_neuron.net)
                d_E_outH = d_E_netO * d_netO_outH
                d_E_netH = d_E_outH * d_outH_netH
                for hidden_weight_i in range(len(hidden_neuron.inputLinks)):
                    d_netH_wi = hidden_neuron.inputLinks[hidden_weight_i].value
                    d_E_wi = d_E_netH * d_netH_wi
                    hidden_neuron.weights[hidden_weight_i] -= LEARNING_RATE * d_E_wi

            for weight_i in range(len(output_neuron.inputLinks)):
                d_netO_wi = output_neuron.inputLinks[weight_i].value
                d_E_wi = d_E_netO * d_netO_wi
                # print(f"d_n1_w{i}: {d_n1_wi}")
                # print(f"d_E_w{i}: {d_E_wi}")
                # print(f"Delta: {-LEARNING_RATE * d_E_wi}")
                output_neuron.weights[weight_i] -= LEARNING_RATE * d_E_wi

        mean_error = np.mean(errors)

        # updating minimal error
        if(mean_error < minimal_error):
            minimal_error = mean_error

        # outputting results
        output(iteration, minimal_error, currentNetwork)

        if check_stop_conditions(minimal_error, weights):
            break

    print()

    # returning best network
    return currentNetwork


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

    # randomly generating data
    # data = []
    # for i in range(1000):
    #     gender = "M" if random.random() < 0.5 else "F"
    #     data.append([str(i), random.random()**3 * 200, random.uniform(0, 200), gender])

    # for i in range(500):
    #     angle = random.random() * 360
    #     radius = random.uniform(50, 100)
    #     x = math.cos(angle * math.pi / 180) * radius + 100
    #     y = math.sin(angle * math.pi / 180) * radius + 100
    #     data.append([str(i), x, y, "M"])
    # for i in range(200):
    #     angle = random.random() * 360
    #     radius = random.uniform(10, 50)
    #     x = math.cos(angle * math.pi / 180) * radius + 100
    #     y = math.sin(angle * math.pi / 180) * radius + 100
    #     data.append([str(i), x, y, "F"])

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
    best_network = train_backprop(list(df["Weight"]), list(df["Height"]), list(df["Gender"]))

    print()

    # printing weights of the best network
    print("Weights")
    for neuron in best_network.hiddenNeurons:
        print(neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))

    print()

    # resulting time
    time2 = time.time()
    print(f"Time: {time2 - time1}")
