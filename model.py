import numpy as np
import csv
import os

def load_data(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        samples, n_input, n_output = map(int, next(reader))
        inputs = np.zeros((samples, n_input))
        outputs = np.zeros((samples, n_output), dtype=bool)
        for sample, line in enumerate(reader):
            inputs[sample] = line[:n_input]
            outputs[sample] = [True if o == '1' else False for o in line[n_input:]]
    return inputs, outputs

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_deriv(x):
    return sig(x) * (1 - sig(x))


class NeuralNetwork:
    def __init__(self, model=None):
        if model:
            self.load_model(model)

    def load_model(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=' ')

            self.n_input, self.n_hidden, self.n_output = map(int, next(reader))
            self.weights = {
                1: np.zeros((self.n_input+1, self.n_hidden)),
                2: np.zeros((self.n_hidden+1, self.n_output))
            }

            for i, line in enumerate(reader):
                if i < self.n_hidden:
                    self.weights[1][:, i] = line
                else:
                    self.weights[2][:, i - self.n_hidden] = line

    def save_model(self, filename):
        with open(filename, 'w+', newline='') as f:
            f.write(' '.join([str(self.n_input), str(self.n_hidden), str(self.n_output)]) + '\n')
            for i in range(self.n_hidden + self.n_output):
                if i < self.n_hidden:
                    row = self.weights[1][:, i]
                else:
                    row = self.weights[2][:, i - self.n_hidden]
                f.write(' '.join(['{0:.3f}'.format(n) for n in row]) + '\n')

    def load_data(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            samples, n_input, n_output = map(int, next(reader))
            inputs = np.zeros((samples, n_input))
            outputs = np.zeros((samples, n_output), dtype=bool)
            for sample, line in enumerate(reader):
                inputs[sample] = line[:n_input]
                outputs[sample] = [True if o == '1' else False for o in line[n_input:]]
        return inputs, outputs

    def train(self, inputs, outputs, epochs=1, alpha=.1):
        for epoch in range(epochs):
            for x, y in zip(inputs, outputs):
                # The pre-activated sum at a node
                layer_input = [
                    np.zeros(self.n_input),
                    np.zeros(self.n_hidden),
                    np.zeros(self.n_output)
                ]

                # Activated node
                activations = [
                    np.zeros(self.n_input),
                    np.zeros(self.n_hidden),
                    np.zeros(self.n_output)
                ]

                # Deltas for backpropagation
                deltas = [
                    np.zeros(self.n_input),
                    np.zeros(self.n_hidden),
                    np.zeros(self.n_output)
                ]

                # Forward propagation
                #  activations[0][0] = -1
                #  for i in range(self.n_input):
                    #  activations[0][i+1] = x[i]
                activations[0] = x

                for l in range(1, 3):
                    layer_input[l] = np.dot(np.insert(activations[l-1], 0, -1), self.weights[l])
                    activations[l] = sig(layer_input[l])
                    #  activations[l][0] = -1
                    #  for j in range(len(layer_input[l])):
                        #  layer_input[l][j] = np.sum(self.weights[l][:, j] * activations[l-1])
                        #  activations[l][j+1] = sig(layer_input[l][j])

                # Back propagation
                deltas[2] = sig_deriv(layer_input[2]) * (y - activations[2])
                #  for j in range(len(layer_input[2])):
                    #  deltas[2][j] = sig_deriv(layer_input[2][j]) * y[j] - activations[2][j]

                deltas[1] = sig_deriv(layer_input[1]) * np.dot(self.weights[2][1:, :], deltas[2])
                #  for i in range(len(layer_input[1])):
                    #  deltas[1][i] = sig_deriv(layer_input[1][i]) * np.sum(self.weights[2][i, :] * deltas[2][i])
                for layer in range(1, 3):
                    for (i, j), value in np.ndenumerate(self.weights[layer]):
                        if i == 0:
                            self.weights[layer][i, j] += alpha * -1 * deltas[layer][j]
                        else:
                            self.weights[layer][i, j] += alpha * activations[layer-1][i-1] * deltas[layer][j]
        return self.weights

if __name__ == '__main__':
    nn = NeuralNetwork('models/sample.NNWDBC.init')
    X, Y = load_data('train/wdbc.mini_train')
    nn.train(X, Y, epochs=1)
    nn.save_model('models/test')



