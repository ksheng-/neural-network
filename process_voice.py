import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('data/voice/voice.csv')
df['label'] = np.where(df['label'] == 'male', 1, 0)
scaler = preprocessing.StandardScaler()
df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
train, test = train_test_split(df, test_size=.4)
n_output = 1
n_input = df.shape[1] - n_output
n_hidden = 5

with open('train/voice.train', 'w+') as f:
    f.write('{} {} {}'.format(train.shape[0], n_input, n_output) + '\n' + train.to_csv(index=False, header=False, sep=' ', float_format='%.3f'))

with open('test/voice.test', 'w+') as f:
    f.write('{} {} {}'.format(test.shape[0], n_input, n_output) + '\n' + test.to_csv(index=False, header=False, sep=' ', float_format='%.3f'))

with open('models/NNVoice.init', 'w+') as f:
    f.write('{} {} {}'.format(n_input, n_hidden, n_output) + '\n')
    f.write(pd.DataFrame(np.random.rand(n_hidden, n_input+1)).to_csv(index=False, header=False, sep=' ', float_format='%.3f'))
    f.write(pd.DataFrame(np.random.rand(n_output, n_hidden+1)).to_csv(index=False, header=False, sep=' ', float_format='%.3f'))
