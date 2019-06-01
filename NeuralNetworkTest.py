import numpy as np
import pandas as pd
import random
import pygame
import sys
import math

POPULATION_SIZE = 10
CROSSOVER_POWER = 2
MUTATION_POWER = 100
MAX_MUTATION = 1
ITERATIONS = 100
MINIMAL_ERROR_SHUTDOWN = False

HIDDEN_LAYER_NEURONS = 8
CLIP_VALUES = False

PRINT_WEIGHTS = False
RENDER_INTERPOLATION_STEP = 5

last_id = 0
def increase_last_id():
	global last_id
	last_id += 1

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return max(0, x)

def tanh(x):
	return (math.tanh(x) + 1) / 2

class Neuron:

	def __init__(self, name, weights, bias):
		self.name = name
		self.weights = weights
		self.bias = bias

	def feedforward(self, inputs):
		total = np.dot(self.weights, inputs) + self.bias
		output = sigmoid(total)
		return output

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
		# 	x[i] = 1.0/x[i]
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

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
    	print("x: " + str(x))
    	print("y: " + str(y))
    	print("points: " + str(points))
    	raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def train(df):

	#creating networks
	generation = []
	for i in range(POPULATION_SIZE):
		new_network = NeuralNetwork()
		generation.append(new_network)

	minimal_error = 1.0

	for iteration in range(ITERATIONS):

		print("Generation " + str(iteration + 1) + " " + str(minimal_error))

		if(PRINT_WEIGHTS):
			for i in range(POPULATION_SIZE):
				print("Network " + str(i) + ":    " + str(generation[i].parent1) + " " + str(generation[i].parent2))
				for j in range(len(generation[i].hidden_neurons)):
					neuron = generation[i].hidden_neurons[j]
					print("    " + neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))

		#calculating error
		network_mean_errors = []
		for i in range(POPULATION_SIZE):
			errors = []
			for j in range(len(df.index)):
				result = generation[i].feedforward([df.loc[j]["Weight"], df.loc[j]["Height"]])
				gender = 0 if df.loc[j]["Gender"] == "M" else 1
				error = abs(result - gender)
				errors.append(error)
			mean_error = np.mean(errors)
			network_mean_errors.append(mean_error)

		#calculating fitness
		for i in range(POPULATION_SIZE):
			generation[i].fitness = 1.0/network_mean_errors[i]

		#list has to be sorted
		generation.sort(key = lambda x: x.fitness, reverse = True)

		#updating minimal error
		if(1.0/generation[0].fitness < minimal_error):
			minimal_error = 1.0/generation[0].fitness

		#creating new generation
		new_generation = []
		for i in range(POPULATION_SIZE):

			#preserving the best network
			if i == 0:
				new_network = crossover(generation[0], generation[0])
				new_network.parent1 = generation[0].id
				new_network.parent2 = generation[0].id
				new_generation.append(new_network)
				continue

			#choosing parents
			rand1 = random.random()**CROSSOVER_POWER;
			rand2 = random.random()**CROSSOVER_POWER;
			scaledRand1 = rand1 * POPULATION_SIZE;
			scaledRand2 = rand2 * POPULATION_SIZE;
			pick1 = int(scaledRand1);
			pick2 = int(scaledRand2);

			#crossover and mutation
			new_network = crossover(generation[pick1], generation[pick2])
			new_network.parent1 = generation[pick1].id
			new_network.parent2 = generation[pick2].id
			new_network.mutate()
			new_generation.append(new_network)

		#swapping generations
		generation = new_generation

		#rendering graph
		best_network = generation[0]
		pixels = pygame.surfarray.pixels2d(surface)
		points = []
		#collecting data points
		for y in range(0, surface.get_height(), RENDER_INTERPOLATION_STEP):
			for x in range(0, surface.get_width(), RENDER_INTERPOLATION_STEP):
				result = best_network.feedforward([x - weight_mean, y - height_mean])
				points.append((x, y, result))
		#drawing interpolated points
		for y in range(surface.get_height() - RENDER_INTERPOLATION_STEP - 1):
			for x in range(surface.get_width() - RENDER_INTERPOLATION_STEP - 1):
				column = int(float(x) / RENDER_INTERPOLATION_STEP)
				row = int(float(y) / RENDER_INTERPOLATION_STEP)
				column_count = int(surface.get_width() / RENDER_INTERPOLATION_STEP)
				point1_index = row*column_count + column
				point2_index = point1_index + 1
				point3_index = (row + 1)*column_count + column
				point4_index = (row + 1)*column_count + column + 1
				# print("x: " + str(x))
				# print("y: " + str(y))
				# print("column: " + str(column) + " " + str(float(x) / surface.get_width() * RENDER_INTERPOLATION_STEP))
				# print("row: " + str(row))
				# print("column_count: " + str(column_count))
				point_list = [points[point1_index], points[point2_index], points[point3_index], points[point4_index]]
				result = bilinear_interpolation(x, y, point_list)
				scaled_result = result*255
				if(scaled_result < 0):
					scaled_result = 0
				if(scaled_result > 255):
					scaled_result = 255
				pixels[x, y] = pygame.Color(0, int(scaled_result), int(scaled_result), int(scaled_result))

		del pixels

		screen.fill((255, 255, 255))
		screen.blit(surface, (0, 0))

		#drawing data points
		for i in range(len(df.index)):
			x = int(df.loc[i]["Weight"] + weight_mean)
			y = int(df.loc[i]["Height"] + height_mean)
			color = pygame.Color(255, 50, 50) if df.loc[i]["Gender"] == "F" else pygame.Color(150, 150, 255)
			pygame.draw.circle(screen, color, (x, y), 1)
			pygame.display.flip()

		#reacting to events, so window can be closed
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

		#if minimal error goes below threshold, training stops
		if MINIMAL_ERROR_SHUTDOWN:
			if minimal_error < 1.0/len(df.index)/2:
				break

	print()
	print("Minimal error: " + str(minimal_error))

	return generation[0]

#setting data
data = 	[
			["Alice", 123, 65, "F"], 
			["Bob", 160, 72, "M"],
			["Charlie", 152, 70, "M"],
			["Diana", 120, 60, "F"],

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
df = pd.DataFrame(data, columns = ["Name", "Weight", "Height", "Gender"])
weight_mean = center_column(df, "Weight")
height_mean = center_column(df, "Height")

#initializing pygame
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
surface = pygame.Surface((200, 200))

#training
best_network = train(df)

print()

#printing weights of the best network
print("Weights")
for neuron in best_network.hidden_neurons:
	print(neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))
print(best_network.o1.name)
for i in range(len(best_network.hidden_neurons)):
	print("    " + best_network.hidden_neurons[i].name + " " + str(best_network.o1.weights[i]))

#testing on original data
print()
print("Original data")
for j in range(len(df.index)):
	result = best_network.feedforward([df.loc[j]["Weight"], df.loc[j]["Height"]])
	result_gender = "M" if result < 0.5 else "F"
	pass_fail_string = "pass" if result_gender == df.loc[j]["Gender"] else "FAIL"
	print(df.loc[j]["Name"] + ": " + f"{result:.3f}" + " (" + str(result) + ")" + " " + pass_fail_string)

print()

#starting render
while(1):
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()