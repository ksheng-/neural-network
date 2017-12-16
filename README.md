# neural-network
a simple neural network implementation

## Setup
Clone the project:
```
git clone https://github.com/ksheng-/neural-network.git
cd neural-network
```
Make sure Python 3.x is installed, and install the dependencies:
```
pip install -r requirements.txt
```

This program was tested on Arch Linux. For other flavors of Linux, you may need to replace all instances of pip with pip3 and python with python3.

## Directories
### data
Contains the raw datasets
### results
Contains example output result files
### train
Contains example training files
### test
Contains example test files
### models
Contains neural network models

## Train a neural network
```
python train.py
```
The program will prompt you for the input files, output file and parameters. If an invalid input is entered, it will re-prompt. ^C will exit the program.

```
python test.py
```
The program will prompt you for the input files, output file, and parameters. If an invalid input is entered, it will re-prompt. ^C will exit the program.

## Dataset
The custom set comes from https://www.kaggle.com/miroslavsabo/young-people-survey, a 2013 survey of Slovakian young people between the ages of 15 and 30. 
The full dataset is 1010 rows and 150 columns, with 139 integer and 11 categorical features. Each of the features is a questionnaire response about various topics as listed below:
* Music preferences (19 items)
* Movie preferences (12 items)
* Hobbies & interests (32 items)
* Phobias (10 items)
* Health habits (3 items)
* Personality traits, views on life, & opinions (57 items)
* Spending habits (7 items)
* Demographics (10 items)
Most integer features range from 1-5, 1 being negative and 5 being positive. The dataset was cleaned using pandas and scikit-learn, in process_youth.py. Since the categorical features don't mean anything to the neural-net, they were split into 1-of-K features using pandas. The output variable was removed, converted to boolean, and readded after processing. The rest of the data was normalized using scikit-learn's MinMaxScaler, and scikit-learn's train_test_split function was used to split the dataset 40/60 into testing and training data. Since the dataset is missing a few values, 0's were imputed for nan's using pandas.

In order to generate weights for the initial neural network, numpy's random.rand function was used, drawing random samples from a uniform distribution between 0 and 1.

For this example, I pulled out the "Drinking" category to be the output
