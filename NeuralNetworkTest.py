import numpy as np
import pandas as pd
import random
import pygame
import sys

POPULATION_SIZE = 10
CROSSOVER_POWER = 2
MUTATION_POWER = 100
MAX_MUTATION = 1000
ITERATIONS = 100

last_id = 0
def increase_last_id():
	global last_id
	last_id += 1

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

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
		self.id = last_id
		increase_last_id()
		self.parent1 = -1
		self.parent2 = -1
		self.h1 = Neuron("h1", [np.random.normal(), np.random.normal()], np.random.normal())
		self.h2 = Neuron("h2", [np.random.normal(), np.random.normal()], np.random.normal())
		self.h3 = Neuron("h3", [np.random.normal(), np.random.normal()], np.random.normal())
		self.o1 = Neuron("o1", [np.random.normal(), np.random.normal(), np.random.normal()], np.random.normal())

	def feedforward(self, x):
		# print(x)
		h1_out = self.h1.feedforward(x)
		h2_out = self.h2.feedforward(x)
		h3_out = self.h3.feedforward(x)
		o1_out = self.o1.feedforward([h1_out, h2_out, h3_out])
		return o1_out

	def mutate(self):
		self.h1.mutate()
		self.h2.mutate()
		self.h3.mutate()
		self.o1.mutate()

def lists_average(list1, list2):
	avg_list = []
	assert len(list1) == len(list2), "Lists have different length"
	for i in range(len(list1)):
		avg_list.append(np.mean([list1[i], list2[i]]))
	return avg_list

def neuron_crossover(neuron1, neuron2):
	return Neuron("New neuron", lists_average(neuron1.weights, neuron2.weights), np.mean([neuron1.bias, neuron2.bias]))

def crossover(network1, network2):
	new_network = NeuralNetwork()
	new_network.h1 = neuron_crossover(network1.h1, network2.h1)
	new_network.h2 = neuron_crossover(network1.h2, network2.h2)
	new_network.h3 = neuron_crossover(network1.h3, network2.h3)
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

	for iteration in range(ITERATIONS):

		print("Generation " + str(iteration + 1))

		for i in range(POPULATION_SIZE):
			print("Network " + str(i) + ":    " + str(generation[i].parent1) + " " + str(generation[i].parent2))
			print("    h1 " + str(generation[i].h1.weights) + " " + str(generation[i].h1.bias))
			print("    h2 " + str(generation[i].h2.weights) + " " + str(generation[i].h2.bias))
			print("    h3 " + str(generation[i].h3.weights) + " " + str(generation[i].h3.bias))
			print("    o1 " + str(generation[i].o1.weights) + " " + str(generation[i].o1.bias))

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
		print()
		for i in range(len(network_mean_errors)):
			print(f"{network_mean_errors[i]:.3f}" + " (" + str(network_mean_errors[i]) + ")")
		print()

		#calculating fitness
		for i in range(POPULATION_SIZE):
			generation[i].fitness = 1.0/network_mean_errors[i]

		#list has to be sorted
		generation.sort(key = lambda x: x.fitness, reverse = True)

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

	return generation[0]

#setting data
data = 	[
			["Alice", 133, 65, "F"], 
			["Bob", 160, 72, "M"],
			["Charlie", 152, 70, "M"],
			["Diana", 120, 60, "F"],
			["Eugene", 164, 69, "M"],
			["Fiona", 149, 65, "F"],
			["Garreth", 177, 75, "M"],
			["Heather", 155, 55, "F"],
			["Short man", 50, 30, "M"]
		]
df = pd.DataFrame(data, columns = ["Name", "Weight", "Height", "Gender"])
weight_mean = center_column(df, "Weight")
height_mean = center_column(df, "Height")

#training
best_network = train(df)

#testing on original data
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
		pixels[i, j] = pygame.Color(0, int(scaled_result), int(scaled_result), int(scaled_result))
del pixels

#starting render
while(1):
	screen.fill((255, 255, 255))
	screen.blit(surface, (0, 0))

	for i in range(len(df.index)):
		x = int(df.loc[i]["Weight"] + weight_mean)
		y = int(df.loc[i]["Height"] + height_mean)
		color = pygame.Color(255, 50, 50) if df.loc[i]["Gender"] == "F" else pygame.Color(150, 150, 255)
		pygame.draw.circle(screen, color, (x, y), 3)

	pygame.display.flip()

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()