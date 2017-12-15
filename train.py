import nn

model = nn.prompt('Initial model file: ', 'model')
x, y = nn.prompt('Training data file: ', 'data')
outfile = input('Output model file: ')
epochs = nn.prompt('Training epochs: ', 'epochs')
rate = nn.prompt('Learning rate: ', 'rate')

print('Training neural network...')
model.train(x, y, epochs, rate)
model.save_model(outfile)
print('Done. Model saved to \'{}\'.'.format(outfile))

