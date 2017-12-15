import nn

model = nn.prompt('Model file: ', 'model')
x, y = nn.prompt('Testing data file: ', 'data')
outfile = input('Results file: ')

print('Testing neural network...')
metrics = model.test(x, y)
nn.save_metrics(metrics, outfile)
print('Done. Results saved to \'{}\'.'.format(outfile))
