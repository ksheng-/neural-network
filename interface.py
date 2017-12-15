import nn

def prompt(message, category):
    while True:
        try:
            val = input(message + '\n' + '--> ')
            if category == 'model':
                return nn.NeuralNetwork(val)
            elif category == 'data':
                return nn.load_data(val)
            elif category == 'epoch':
                return int(val)
            elif category == 'rate':
                return float(val)
        except ValueError:
            continue
