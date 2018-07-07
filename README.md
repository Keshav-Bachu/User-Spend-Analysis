# User-Spend-Analysis
A neural network program to predict if a person will spend money on a given application

## Overview
This project used data collected from an online community to learn and predict spending habits based on specific inputs. The inputs/questions posed to users are found within the csv files.
Since there were only ~380 points of data, the project uses a 90 - 10 split for test and dev respectively. 

Training on this data gave the following results:

Train Set Accuracy    - 92.19219%

Dev Set Accuracy      - 91.89189%

This is a good start and more data would be helpful in letting this model grow some more allowing it to work with high accuracy on larger input data along with different types of people as an online community represents only a fraction of the the
type of people who use the application.


## Files
FEHUserAnalysis.csv   - CSV file used to load in the numpy data

FormatData.py         - Python file run as the 'main' and used to format data from the csv file and pass it to the NN

Key Indications.txt.   - Text file that writes out how the data was converted form the string format in the survey to numerical inputs

PredictUserSpending.py- Python file with the NN model used for this project (written in tensorflow)

*.npy files           - Trained weights that can be used to predict on more data (instructions to load and use them within the code)

## Built With
Numpy                       - Used to store weights and format data

Tensorflow-gpu/Tensorflow   - Used to build the model and train weights

## Prerequisites
Appropriate packages used within the project (most importantly Numpy and Tensorflow/Tensorflow-gpu)

Python 3.6.5 or above (project built on 3.6.5, will most likely work with older versions, but recommended to use the newest version)


## Usage/Deployment
1.  Clone this repo to your local system

2.  Open the FormatData.py file and run said file, the code will handel the rest

## Author
* Keshav Bachu
