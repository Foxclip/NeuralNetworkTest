import numpy as np

def sigmoid(x):
	# Our activation function: f(x) = 1 / (1 + e^(-x))
	return 1 / (1 + np.exp(-x))

def mse_loss(y_true, y_pred):
	return ((y_true - y_pred)**2).mean()

class Neuron:
	def __init__(self, name, weights, bias):
		self.name = name
		self.weights = weights
		self.bias = bias

	def feedforward(self, inputs):
		# Weight inputs, add bias, then use the activation function
		print(self.name + " inputs: " + str(inputs))
		total = np.dot(self.weights, inputs) + self.bias
		output = sigmoid(total)
		print(self.name + " output: " + str(output))
		return output

class NeuralNetwork:
	def __init__(self):
		weights = np.array([1, 1])
		bias = 0

		self.h1 = Neuron("h1", weights, bias)
		self.h2 = Neuron("h2", weights, bias)
		self.o1 = Neuron("o1", weights, bias)

	def feedforward(self, x):
		out_h1 = self.h1.feedforward(x)
		out_h2 = self.h2.feedforward(x)
		out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

		return out_o1

network = NeuralNetwork()
x = np.array([-2, -1])
print(network.feedforward(x))

# y_true = np.array([1, 0, 0, 1])
# y_pred = np.array([0, 0, 0, 0])
# print(mse_loss(y_true, y_pred))