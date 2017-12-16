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

def save_metrics(metrics, filename):
    with open(filename, 'w+', newline='') as f:
        class_confusion, class_metrics, microaverage_metrics, macroaverage_metrics = metrics
        for i in range(class_confusion.shape[0]):
            f.write(' '.join(['{}'.format(n) for n in class_confusion[i]]) + ' ')
            f.write(' '.join(['{0:.3f}'.format(n) for n in class_metrics[i]]) + '\n')
        f.write(' '.join(['{0:.3f}'.format(n) for n in microaverage_metrics[0]]) + '\n')
        f.write(' '.join(['{0:.3f}'.format(n) for n in macroaverage_metrics]) + '\n')
        
def prompt(message, category):
    while True:
        try:
            val = input(message + '\n' + '--> ')
            if category == 'model':
                return NeuralNetwork(val)
            elif category == 'data':
                return load_data(val)
            elif category == 'epochs':
                return int(val)
            elif category == 'rate':
                if 0 < float(val) <= 1:
                    return float(val)
                else:
                    continue
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            continue

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
                activations[0] = x

                for l in range(1, 3):
                    layer_input[l] = np.dot(np.insert(activations[l-1], 0, -1), self.weights[l])
                    activations[l] = sig(layer_input[l])

                # Back propagation
                deltas[2] = sig_deriv(layer_input[2]) * (y - activations[2])
                deltas[1] = sig_deriv(layer_input[1]) * np.dot(self.weights[2][1:, :], deltas[2])

                for layer in range(1, 3):
                    for (i, j), value in np.ndenumerate(self.weights[layer]):
                        if i == 0:
                            self.weights[layer][i, j] += alpha * -1 * deltas[layer][j]
                        else:
                            self.weights[layer][i, j] += alpha * activations[layer-1][i-1] * deltas[layer][j]
        return self.weights

    def _accuracy(self, confusion):
        a, b, c, d = np.hsplit(confusion, 4)
        return (a + d) / (a + b + c + d)
    
    def _precision(self, confusion):
        a, b, c, d = np.hsplit(confusion, 4)
        return a / (a + b)
    
    def _recall(self, confusion):
        a, b, c, d = np.hsplit(confusion, 4)
        return a / (a + c)
    
    def _f1(self, precision, recall):
        return (2 * precision * recall) / (precision + recall)
    
    def metrics(self, confusion):
        precision = self._precision(confusion)
        recall = self._recall(confusion)
        metrics = np.concatenate((
            self._accuracy(confusion),
            precision,
            recall,
            self._f1(precision, recall)
            ), axis=1
        )
        return metrics

    def test(self, inputs, outputs):
        n_samples, n_classes = outputs.shape[0], outputs.shape[1]
        confusion = np.zeros((n_samples, n_classes, 4), dtype=bool)
        for i, (x, y) in enumerate(zip(inputs, outputs)):
            activations = x
            for l in range(1, 3):
                activations = sig(np.dot(np.insert(activations, 0, -1), self.weights[l]))
            # TODO: Non floating point rounding?
            output = np.where(activations >= .5, 1, 0)
            for node, (predicted, expected) in enumerate(zip(output, y)):
                if predicted and expected:
                    confusion[i][node][0] = True
                elif predicted and not expected:
                    confusion[i][node][1] = True
                elif expected and not predicted:
                    confusion[i][node][2] = True
                else:
                    confusion[i][node][3] = True
        
        class_confusion = np.reshape(np.sum(confusion, axis=0), (n_classes, 4))
        class_metrics = self.metrics(class_confusion)    
        microaverage_metrics = self.metrics(np.sum(class_confusion, axis=0, keepdims=True))
        macroaverage_metrics = np.mean(class_metrics, axis=0)
        macroaverage_metrics[3] = self._f1(macroaverage_metrics[1], macroaverage_metrics[2])
        return (class_confusion, class_metrics, microaverage_metrics, macroaverage_metrics)

if __name__ == '__main__':
    nn = NeuralNetwork('models/NNVoice.init')
    X, Y = load_data('train/voice.train')
    nn.train(X, Y, epochs=100, alpha=.1)
    nn.save_model('models/NNVoice.1.100.trained')

    X, Y = load_data('test/voice.test')
    trained = NeuralNetwork('models/NNVoice.1.100.trained')
    metrics = trained.test(X, Y)
    save_metrics(metrics, 'results/voice.1.100.results')
