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



