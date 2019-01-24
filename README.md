# Music-Genre-Classification-SVM-ANN-CNN

README

Program: Music genre Classification
Performs classification of a “music files in au format”, trains a classifier parameters using training data and then can be used for classification of new unclassified data.

Program Description: 
Takes input training music files in .au format and validation files in same format. Estimates the parameters using the training data to be used for classification of the validation data. 

Running The Program:
We have provided some python scripts, whose name represent their functionality and the classifier used. These files can be run from command line in any operating system.
To run the script first make the folder with the python scripts as the current folder. Also, make sure that both the training file (genre) and validation file (validation) are present in the same folder.

Requirements:
Needs python version 3.6 or higher available to run the script.

Has dependency on the following python modules:

pandas ( to read csv files)
numpy (for array manipulation)
math (to calculate logarithmic values)
csv (to write into csv files)
matplotlib (for plotting)
sklearn.model_selection (for selecting classififer model)
librosa (for music feature extraction)
seaborn (for plotting)
sklearn.preprocessing (for )
sklearn.neural_network (to import neural module)
sklearn.metrics (to calculate accuracy)
fnmatch (for filename preprocessing)


python scripts:

1. 2D_MELSPECTROGRAM_CNN

2. FFT_ANN

3. FFT_SVM

4. MFCC_AVG_SVM

5. MFCC_AVG_ANN

6. MFCC_AVG_STD_SVM

7. MFCC_AVG__STD_ANN

8. MFCC_AVG_STD_OTHERS_SVM

9. MFCC_AVG_STD_OTHERS_ANN

10.OSC_SVM

11.OSC_ANN

12.OSC_OTHERS_SVM

13. OSC_OTHERS_MLP

14. OSC_OTHERS_MFCC_SVM

15. OSC_OTHERS_MFCC_ANN





Usage:
An example command line input to run the script:

python OSC_OTHERS_SVM.py

Here first argument ‘python’ asks to use python for compiling, ‘logistic_regression.py’ is the python script name.

Usage Options:
A generic command line input to run the script:

python script_name

script_name: name of the python script (here, ‘OSC_OTHERS_SVM.py’)

Built With:

Python version 3.6.2
