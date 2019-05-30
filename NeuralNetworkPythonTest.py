import numpy as np
import pandas as pd
import random

POPULATION_SIZE = 2
CROSSOVER_POWER = 2
MUTATION_POWER = 100
MAX_MUTATION = 0

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

	def mutate(self, mutation_rate):
		for i in range(len(self.weights)):
			self.weights[i] += random.uniform(-mutation_rate, mutation_rate)
			# self.weights[i] = np.clip(self.weights[i], -1.0, 1.0)

class NeuralNetwork:

	global last_id
	def __init__(self):
		self.id = last_id
		increase_last_id()
		self.h1 = Neuron("h1", [np.random.normal(), np.random.normal()], np.random.normal())
		self.h2 = Neuron("h2", [np.random.normal(), np.random.normal()], np.random.normal())
		self.o1 = Neuron("o1", [np.random.normal(), np.random.normal()], np.random.normal())
		# self.h1 = Neuron("h1", [1, 1], 0)
		# self.h2 = Neuron("h2", [1, 1], 0)
		# self.o1 = Neuron("o1", [1, 1], 0)

	def feedforward(self, x):
		# print(x)
		h1_out = self.h1.feedforward(x)
		h2_out = self.h2.feedforward(x)
		o1_out = self.o1.feedforward([h1_out, h2_out])
		return o1_out

	def mutate(self, mutation_rate):
		self.h1.mutate(mutation_rate)
		self.h2.mutate(mutation_rate)
		self.o1.mutate(mutation_rate)

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
	new_network.o1 = neuron_crossover(network1.o1, network2.o1)
	return new_network


#setting data
data = [["Alice", 133, 65, "F"], ["Bob", 160, 72, "M"], ["Charlie", 152, 70, "M"], ["Diana", 120, 60, "F"]]
df = pd.DataFrame(data, columns = ["Name", "Weight", "Height", "Gender"])
mean_weight = np.mean(list(df["Weight"]))
mean_height = np.mean(list(df["Height"]))
for i in range(len(df.index)):
	df.iloc[i, df.columns.get_loc("Weight")] -= mean_weight
	df.iloc[i, df.columns.get_loc("Height")] -= mean_height

#creating networks
generation = []
for i in range(POPULATION_SIZE):
	new_network = NeuralNetwork()
	generation.append(new_network)

for i in range(POPULATION_SIZE):
	print("Network " + str(i))
	print("    h1 " + str(generation[i].h1.weights) + " " + str(generation[i].h1.bias))
	print("    h2 " + str(generation[i].h2.weights) + " " + str(generation[i].h2.bias))
	print("    o1 " + str(generation[i].o1.weights) + " " + str(generation[i].o1.bias))

#calculating error
network_mean_errors = []
for i in range(POPULATION_SIZE):
	# print("Network " + str(i))
	errors = []
	for j in range(len(df.index)):
		result = generation[i].feedforward([df.loc[j]["Weight"], df.loc[j]["Height"]])
		gender = 0 if df.loc[j]["Gender"] == "M" else 1
		error = abs(result - gender)
		errors.append(error)
		# print("    " + str(result) + " " + str(error))
	mean_error = np.mean(errors)
	network_mean_errors.append(mean_error)
	# print("    Mean error: " + str(mean_error))
print()
print(network_mean_errors)
print()

#calculating fitness
for i in range(POPULATION_SIZE):
	generation[i].fitness = 1.0 if network_mean_errors[i] == 0 else 1.0/network_mean_errors[i]

#list has to be sorted
generation.sort(key = lambda x: x.fitness, reverse=True)

#creating new generation
new_generation = []
for i in range(POPULATION_SIZE):

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
	new_network.mutate(random.random()**MUTATION_POWER * MAX_MUTATION)
	new_generation.append(new_network)

for i in range(POPULATION_SIZE):
	print("Network " + str(i) + ":    " + str(new_generation[i].parent1) + " " + str(new_generation[i].parent2))
	print("    h1 " + str(new_generation[i].h1.weights) + " " + str(new_generation[i].h1.bias))
	print("    h2 " + str(new_generation[i].h2.weights) + " " + str(new_generation[i].h2.bias))
	print("    o1 " + str(new_generation[i].o1.weights) + " " + str(new_generation[i].o1.bias))