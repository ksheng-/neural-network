import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#  school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
#  sex - student's sex (binary: 'F' - female or 'M' - male)
#  age - student's age (numeric: from 15 to 22)
#  address - student's home address type (binary: 'U' - urban or 'R' - rural)
#  famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
#  Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
#  Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
#  Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
#  Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
#  Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
#  reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
#  guardian - student's guardian (nominal: 'mother', 'father' or 'other')
#  traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
#  studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
#  failures - number of past class failures (numeric: n if 1<=n<3, else 4)
#  schoolsup - extra educational support (binary: yes or no)
#  famsup - family educational support (binary: yes or no)
#  paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
#  activities - extra-curricular activities (binary: yes or no)
#  nursery - attended nursery school (binary: yes or no)
#  higher - wants to take higher education (binary: yes or no)
#  internet - Internet access at home (binary: yes or no)
#  romantic - with a romantic relationship (binary: yes or no)
#  famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
#  freetime - free time after school (numeric: from 1 - very low to 5 - very high)
#  goout - going out with friends (numeric: from 1 - very low to 5 - very high)
#  Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
#  Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
#  health - current health status (numeric: from 1 - very bad to 5 - very good)
#  absences - number of school absences (numeric: from 0 to 93)

df = pd.read_csv('data/student-alcohol-consumption/student-mat.csv')
is_alcoholic = np.where(df['Dalc'] + df['Walc'] > 4, 1, 0)
df = df.drop(['Dalc', 'Walc', 'school'], axis=1)
#  df = pd.get_dummies(df)
df = pd.get_dummies(df, columns=['Mjob', 'Fjob', 'reason', 'guardian'])
df['sex'] = np.where(df['sex'] == 'F', 0, 1)
df['address'] = np.where(df['address'] == 'U', 0, 1)
df['famsize'] = np.where(df['famsize'] == 'LE3', 0, 1)
df['Pstatus'] = np.where(df['Pstatus'] == 'T', 0, 1)
df = df.replace('yes', 1)
df = df.replace('no', 0)

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df.iloc[:, :-1])
df.iloc[:, :-1] = np_scaled

df['is_alcoholic'] = is_alcoholic
print(df)
train, test = train_test_split(df, test_size=.5)

n_output = 1
n_input = df.shape[1] - n_output
n_hidden = 5
with open('train/alcohol.train', 'w+') as f:
    f.write('{} {} {}'.format(df.shape[0], n_input, n_output) + '\n' + train.to_csv(index=False, header=False, sep=' ', float_format='%.3f'))

with open('test/alcohol.test', 'w+') as f:
    f.write('{} {} {}'.format(df.shape[0], n_input, n_output) + '\n' + test.to_csv(index=False, header=False, sep=' ', float_format='%.3f'))

with open('models/NNAlcohol.init', 'w+') as f:
    f.write('{} {} {}'.format(n_input, n_hidden, n_output) + '\n')
    f.write(pd.DataFrame(np.random.rand(n_hidden, n_input+1)).to_csv(index=False, header=False, sep=' ', float_format='%.3f'))
    f.write(pd.DataFrame(np.random.rand(n_output, n_hidden+1)).to_csv(index=False, header=False, sep=' ', float_format='%.3f'))
