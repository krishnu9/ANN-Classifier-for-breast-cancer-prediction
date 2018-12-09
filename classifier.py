# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

#Load dataset 
dataset = pd.read_csv('BreastCancerDataset.csv')
dataset = dataset.replace('?',np.NaN)   #Reaplce '?' with NaN(value mising)
x = dataset.iloc[:, 1:10].values
Y = dataset.iloc[:, 10].values
y = Y==4

#Replace missing values with mean of the column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 1:10])
x[:, 1:10] = imputer.transform(x[:, 1:10])

#Split data into train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense

#Initialise the ANN classifier
classifier = Sequential()

#Add the input layer and 1st hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 9))
#Add second hidden layer
classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
#Add output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

#Fit the ANN on training data
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 50)