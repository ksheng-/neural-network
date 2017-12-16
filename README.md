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
#### data
Contains the raw datasets
#### results
Contains example output result files
#### train
Contains example training files
#### test
Contains example test files
#### models
Contains neural network models

## Train a neural network
```
python train.py
```
The program will prompt you for the initial neural network file, the training data, a network output file and the learning parameters. If an invalid input is entered, it will re-prompt. ^C will exit the program. The neural net is trained on the data and the updated weights are saved in the given file.

```
python test.py
```
The program will prompt you for a neural network, training data, and a results output file. If an invalid input is entered, it will re-prompt. ^C will exit the program. The data is run through the neural network, and the accuracy metrics are saved to the results file.

## Example Dataset
The custom set comes from https://www.kaggle.com/primaryobjects/voicegender, a dataset of voice properties labeled by sex.
The full dataset is 3168 rows and 21 columns, with 20 floating plot features and a label. The features are listed below:
*  meanfreq: mean frequency (in kHz)
*  sd: standard deviation of frequency
*  median: median frequency (in kHz)
*  Q25: first quantile (in kHz)
*  Q75: third quantile (in kHz)
*  IQR: interquantile range (in kHz)
*  skew: skewness (see note in specprop description)
*  kurt: kurtosis (see note in specprop description)
*  sp.ent: spectral entropy
*  sfm: spectral flatness
*  mode: mode frequency
*  centroid: frequency centroid (see specprop)
*  peakf: peak frequency (frequency with highest energy)
*  meanfun: average of fundamental frequency measured across acoustic signal
*  minfun: minimum fundamental frequency measured across acoustic signal
*  maxfun: maximum fundamental frequency measured across acoustic signal
*  meandom: average of dominant frequency measured across acoustic signal
*  mindom: minimum of dominant frequency measured across acoustic signal
*  maxdom: maximum of dominant frequency measured across acoustic signal
*  dfrange: range of dominant frequency measured across acoustic signal
*   modindx: modulation index. Calculated as the accumulated absolute difference between adjacent measurements of fundamental frequencies divided by the frequency range
*  label: male or female

The raw data can be found in data/voice/voice.csv, the the parsing script is process_voice.py.

The dataset was cleaned using pandas and scikit-learn. The 'male/female' output variable was converted to boolean, and readded after processing. The rest of the data was normalized using scikit-learn's StandardScaler, and scikit-learn's train_test_split function was used to split the dataset 40/60 into testing and training data. 

In order to generate weights for the initial neural network, numpy's random.rand function was used, drawing random samples from a uniform distribution between 0 and 1.

The example training set can be found in train/voice.train, and the testing set in test/voice.test. The initial network weights can be found in models/NNVoice.init.

After training for 100 epochs with a learning rate of .1 and 5 hidden nodes, the network achieved an f1 score of .970 and an accuracy of .969. The network can be found in models/voice.1.100.trained and the results in results/voice.1.100.results. Adjusting the number of hidden nodes or learning parameters did not significantly alter the performance.
