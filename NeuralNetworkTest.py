import numpy as np
import pandas as pd
import random
import pygame
import sys
import math

POPULATION_SIZE = 10
CROSSOVER_POWER = 2
MUTATION_POWER = 100
MAX_MUTATION = 1000
ITERATIONS = 1000

HIDDEN_LAYER_NEURONS = 3

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
		# 	x[i] = x[i]**3
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

def train(df):

	#creating networks
	generation = []
	for i in range(POPULATION_SIZE):
		new_network = NeuralNetwork()
		generation.append(new_network)

	minimal_error = 1.0

	for iteration in range(ITERATIONS):

		print("Generation " + str(iteration + 1))

		# for i in range(POPULATION_SIZE):
		# 	print("Network " + str(i) + ":    " + str(generation[i].parent1) + " " + str(generation[i].parent2))
		# 	for j in range(len(generation[i].hidden_neurons)):
		# 		neuron = generation[i].hidden_neurons[j]
		# 		print("    " + neuron.name + " " + str(neuron.weights) + " " + str(neuron.bias))

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
		# print()
		# for i in range(len(network_mean_errors)):
		# 	print(f"{network_mean_errors[i]:.3f}" + " (" + str(network_mean_errors[i]) + ")")
		# print()

		#calculating fitness
		for i in range(POPULATION_SIZE):
			generation[i].fitness = 1.0/network_mean_errors[i]

		#list has to be sorted
		generation.sort(key = lambda x: x.fitness, reverse = True)
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

		#cheking if best network passes all tests
		fail = False
		for i in range(len(df.index)):
			result = generation[0].feedforward([df.loc[i]["Weight"], df.loc[i]["Height"]])
			result_gender = "M" if result < 0.5 else "F"
			if result_gender != df.loc[i]["Gender"]:
				fail = True
				break
		if not fail:
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
			["Eugene", 164, 69, "M"],
			["Fiona", 129, 65, "F"],
			["Garreth", 177, 75, "M"],
			["Heather", 135, 55, "F"],
			["Short man 1", 75, 30, "M"],
			["Short man 2", 70, 25, "M"],
			["Short man 3", 80, 28, "M"],
			["Short man 4", 90, 50, "M"],
		]
df = pd.DataFrame(data, columns = ["Name", "Weight", "Height", "Gender"])
weight_mean = center_column(df, "Weight")
height_mean = center_column(df, "Height")

#training
best_network = train(df)

#testing on original data
print()
print("Original data")
for j in range(len(df.index)):
	result = best_network.feedforward([df.loc[j]["Weight"], df.loc[j]["Height"]])
	result_gender = "M" if result < 0.5 else "F"
	pass_fail_string = "pass" if result_gender == df.loc[j]["Gender"] else "FAIL"
	print(df.loc[j]["Name"] + ": " + f"{result:.3f}" + " (" + str(result) + ")" + " " + pass_fail_string)

print()

# #setting new data
# data = [["Eugene", 164, 69, "M"], ["Fiona", 149, 65, "F"], ["Garreth", 177, 75, "M"], ["Heather", 155, 55, "F"]]
# df = pd.DataFrame(data, columns = ["Name", "Weight", "Height", "Gender"])
# center_column(df, "Weight")
# center_column(df, "Height")

# #testing on new data
# print("New data")
# for j in range(len(df.index)):
# 	result = best_network.feedforward([df.loc[j]["Weight"], df.loc[j]["Height"]])
# 	result_gender = "M" if result < 0.5 else "F"
# 	pass_fail_string = "pass" if result_gender == df.loc[j]["Gender"] else "FAIL"
# 	print(df.loc[j]["Name"] + ": " + f"{result:.3f}" + " (" + str(result) + ")" + " " + pass_fail_string)

#initializing pygame
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

pygame.draw.circle(screen, pygame.Color(0, 255, 0, 0), (0, 0), 200)

#rendering graph
surface = pygame.Surface((200, 200))
pixels = pygame.surfarray.pixels2d(surface)
for i in range(surface.get_width()):
	for j in range(surface.get_height()):
		result = best_network.feedforward([i - weight_mean, j - height_mean])
		scaled_result = result*255
		if(scaled_result < 0):
			scaled_result = 0
		if(scaled_result > 255):
			scaled_result = 255
		pixels[i, j] = pygame.Color(0, int(scaled_result), int(scaled_result), int(scaled_result))
del pixels

#starting render
while(1):
	screen.fill((255, 255, 255))
	screen.blit(surface, (0, 0))

	#drawing data points
	for i in range(len(df.index)):
		x = int(df.loc[i]["Weight"] + weight_mean)
		y = int(df.loc[i]["Height"] + height_mean)
		color = pygame.Color(255, 50, 50) if df.loc[i]["Gender"] == "F" else pygame.Color(150, 150, 255)
		pygame.draw.circle(screen, color, (x, y), 3)

	pygame.display.flip()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()