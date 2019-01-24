# import all necessary modules 

import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import csv
from sklearn import svm
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
import os
from fnmatch import fnmatch
from sklearn.model_selection import cross_val_score
import copy 

print ('start')

# read train data files

# training data directory
train_root = 'E:\ML_PROJECT3\genres'

# data type extension
pattern = "*.au"

sdir_no=0
class_dict={}
file_list_label=[]

# walk through training directory and read all .au file names
for path, subdirs, files in os.walk(train_root):
    if (len(subdirs)>1):
        for sd in subdirs:
            
            # create a dictionary of genre folder name
            # and numeric value label
            
            # give each genre a unique numeric value.
            # we start from zero, and
            # each time we encounter a new genre folder name
            # we increase the label counter by one.
            # and give the corresponding genre the current label

            
            if sd not in class_dict:
                class_dict[sd]=sdir_no
                sdir_no += 1
    
    cur_dir=os.path.basename(path)
    cur_label=class_dict.get(cur_dir," ")
    
    for name in files:
        if fnmatch(name, pattern):
            
            filename=os.path.join(path, name)
            
            cur_tuple=(filename,cur_label)
            
            # append training filenames and the corresponding
            # numeric label in list
            file_list_label.append(cur_tuple)
            

# inverse the genre name vs numeric label
# and store in a dictionary
            
inverse_CC={}        
for key,value in class_dict.items():
    inverse_CC[value]=key
    
label_list=[]
for key,value in class_dict.items():
    label_list.append(key)
    
    
# validation data directory
validation_root = 'E:\ML_PROJECT3\AA'

#data type extension
pattern = "*.au"

validation_file_list=[]

# walk through validation directory and read all .au file name
for path, subdirs, files in os.walk(validation_root):
    for name in files:
        if fnmatch(name, pattern):
            
            filename=os.path.join(path, name)
            
            # append validation filenames in list
            validation_file_list.append(filename)
            
            

# define hop length
hop_length=512

# create an empty array to store all training data
X_ALL=np.zeros((1,100))

# create an empty list to append all training file labels
labels_ALL = []

# iterate through filename list and read file data
# from file data extract MFCC feature
for name_label in file_list_label:
    
    # read data and sampling rate
    XM,SR=librosa.core.load(name_label[0])
    
    # extract MFFC feature; no of co-efficients
    mfcc = librosa.feature.mfcc(y=XM, sr=SR, hop_length=hop_length, n_mfcc=100)
    
    # take average of MFCC co-efficients across all frames
    mfcc_avg=np.average(mfcc,axis=1)
    
    # reshapoe into a row vector
    mfcc_avg=mfcc_avg.reshape(1,-1)
    
    # append current file's feature vector into
    # train data array
    X_ALL=np.vstack((X_ALL,mfcc_avg))
    
    # read current file label
    temp2=name_label[1]
    
    # append to train data label list
    labels_ALL.append(temp2)


# remove first row of the array as it was garbaze
# to initialize the array
X_ALL=X_ALL[1:,:]

# split all training data into train and validation set
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_ALL, labels_ALL, 
                                                stratify = labels_ALL, random_state = None)


# normalize each column (each feature vector)
scaler=MinMaxScaler()

# fit a normalizer using training data and
# then perform normalization on training data
X_tr2=scaler.fit_transform(X_tr)

# normalize validation data
X_vld2=scaler.transform(X_vld)


# use grid search to find optimal parameters for SVM classifier

# give possible values to iterate over

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
 
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
clf_grid.fit(X_tr2, lab_tr)
 
# print out the best parameters
print("Best Parameters:\n", clf_grid.best_params_)
#print("Best Estimators:\n", clf_grid.best_estimator_)


# train a SVM model using the best parameter
# on the training data 
clf = svm.SVC(kernel='rbf', C = clf_grid.best_params_['C'],\
              gamma=clf_grid.best_params_['gamma']) 

clf.fit(X_tr2, lab_tr)


# perform prediction on the validation data
y_pred = clf.predict(X_vld2)


# calculate accuracy on validation prediction

# print validation accuracy
print("Validation Accuracy:",metrics.accuracy_score(lab_vld, y_pred))

# Compute confusion matrix       
# convert numeric labels to actual label
        
lab_vld_name=[inverse_CC[p] for p in lab_vld]
y_pred_name=[inverse_CC[p] for p in y_pred]

# calculate the confusion matrix
conf_mat=metrics.confusion_matrix(lab_vld_name, y_pred_name,labels=label_list)

# convert into a pandas dataframe

df_cm = pd.DataFrame(conf_mat, index = [i for i in label_list],
                  columns = [i for i in label_list])

# plt the confusion matrix figure
plt.figure(figsize = (15,15))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16},cmap="Blues")
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.show()

# calculate 10 fold validation accuracy


# deepcopy all training data
X_ALL_CV=copy.deepcopy(X_ALL)
# normalize each column (each feature vector)
scaler2=MinMaxScaler()

# fit a normalizer using training data and
# then perform normalization on training data
X_ALL_CV=scaler2.fit_transform(X_ALL_CV)
clf2 = svm.SVC(kernel='rbf', C = clf_grid.best_params_['C'],\
              gamma=clf_grid.best_params_['gamma']) 
scores = cross_val_score(clf2, X_ALL_CV, labels_ALL, cv=10)

# calculate mean accuracy
mean_accuracy=np.mean(scores)

# print mean_accuracy over 10 fold

print (mean_accuracy)
# create an empty array to store all training data
X_VALID=np.zeros((1,100))

# iterate through filename list and read file data
# from file data extract MFCC feature
for name in validation_file_list:
    
    # read data and sampling rate
    XM,SR=librosa.core.load(name)
    
    # extract MFFC feature; no of co-efficients
    mfcc = librosa.feature.mfcc(y=XM, sr=SR, hop_length=hop_length, n_mfcc=100)
    
    # read data and sampling rate
    mfcc_avg=np.average(mfcc,axis=1)
    
    # reshapoe into a row vector
    mfcc_avg=mfcc_avg.reshape(1,-1)
    
    # append current file's feature vector into
    # train data array
    X_VALID=np.vstack((X_VALID,mfcc_avg))


# remove first row of the kaggle_validation array as it was garbaze
# to initialize the array
X_VALID=X_VALID[1:,:]

# transform kaggle_validation data using
# min-max value of training data set
X_test2=scaler.transform(X_VALID)



# perform prediction on validation data
y_valid = clf.predict(X_test2)

# write prediction into a CSV file

# output CSV file name
output_file='SVM_MFCC_AVG_result.csv'

# appending the test data sample id and predicted data sample class label in a list
table=[]
ipd=str('id')
cc=str('class')
table.append([ipd,cc]) 
for i,name in enumerate(validation_file_list):
    y=inverse_CC[y_valid[i]]
    j=os.path.basename(name)
    table.append([j,y])


# creating a csv file of data sample id and their corresponding  class label 
with open(output_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in table:
        writer.writerow(val)