import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('data/young-people-survey/responses.csv')
df = df.iloc[:, 73:140]
is_alcoholic = np.where(df['Alcohol'] == 'drink a lot', 1, 0)
df = df.drop(['Alcohol'], axis=1)
#  df = pd.get_dummies(df)
df = pd.get_dummies(df, columns=['Smoking', 'Punctuality', 'Lying', 'Internet usage'])
#  df = pd.get_dummies(df, columns=['Smoking', 'Alcohol', 'Lying', 'Internet usage', 'Gender', 'Left - right handed', 'Education', 'Only child', 'Village - town', 'House - block of flats'])
#  df['Pstatus'] = np.where(df['Pstatus'] == 'T', 0, 1)
print(np.count_nonzero(is_alcoholic))
scaler = preprocessing.StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df.fillna(0)))

df['is_alcoholic'] = is_alcoholic
train, test = train_test_split(df, test_size=.4)

n_output = 1
n_input = df.shape[1] - n_output
print(n_input)
n_hidden = 20
with open('train/youth.train', 'w+') as f:
    f.write('{} {} {}'.format(train.shape[0], n_input, n_output) + '\n' + train.to_csv(index=False, header=False, sep=' ', float_format='%.3f'))

with open('test/youth.test', 'w+') as f:
    f.write('{} {} {}'.format(test.shape[0], n_input, n_output) + '\n' + test.to_csv(index=False, header=False, sep=' ', float_format='%.3f'))

with open('models/NNYouth.init', 'w+') as f:
    f.write('{} {} {}'.format(n_input, n_hidden, n_output) + '\n')
    f.write(pd.DataFrame(np.random.rand(n_hidden, n_input+1)).to_csv(index=False, header=False, sep=' ', float_format='%.3f'))
    f.write(pd.DataFrame(np.random.rand(n_output, n_hidden+1)).to_csv(index=False, header=False, sep=' ', float_format='%.3f'))
