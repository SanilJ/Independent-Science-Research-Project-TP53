# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense 
from keras.utils.vis_utils import plot_model

#gets the data into the Dataframe 
dataset = pd.read_csv('DatasetV2 - IntegarEncodingStyle.csv')
X = dataset.iloc[:, 0:2].values
Y = dataset.iloc[:, 1].values   

#change this for model tweaking                 
labelencoder_X_1 = LabelEncoder()
X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0]) 
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_2.fit_transform(X[:, 1]) 

 
#splitting the dataset into the Training set and Test set 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)

#initializing Neural Netowork 
classifier = Sequential()

# Adding the input layer and the first hidden layer
# ReLu Activation Layer 
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 2))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Sigmoid Output Layer 
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#epochs are the number of iterations 
# Fitting our model 
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 150)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

print(classifier.summary()) 