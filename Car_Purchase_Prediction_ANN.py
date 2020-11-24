# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 16:08:43 2020

@author: P795864
"""

#Artificial Neural Networks

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
tf.__version__

# Importing the dataset
dataset = pd.read_csv('Car_Purchasing_Data.csv', encoding = 'ISO-8859-1')#we use encoding bcz there are lot of special characters in file, otherwise it throws an utf-8 error
#X = dataset.iloc[:, 3:-1].values
#y = dataset.iloc[:, -1].values

#We can dataset in above format or shown below as well
X = dataset.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1 )    
y = dataset['Car Purchase Amount']
y.shape
y = y.values.reshape(-1,1) # we hvae to reshape the value into array to (500,1) for scaling purpose
y.shape
dataset.head(5)
sns.pairplot(dataset)#------Its visualize the data


"""# Encoding categorical data

# Label Encoding the Gender Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

print(X)

#One Hot Encoding the Geography Column
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

print(X)

"""

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
y_scaled = sc.fit_transform(y)




# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, random_state = 0)


#Building the ANN

#Initializing the ANN
ann = tf.keras.models.Sequential()

#Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 25, activation = 'relu'))#---Units is number of neurons to add

#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 25, activation = 'relu'))

#Adding the Output Layer------ Units = 1 as output is binary i.e one dimensional
ann.add(tf.keras.layers.Dense(units = 1, activation = 'linear')) #--activation function is linear bcz we hv to predict the values

#Training the ANN

#compiling the ANN ----
#Optimizer - To update the weights and reduce the loss error between predictions and real result (we use adam optimiser for Gradient Descent)
#loss - which is weigh to compute the difference between the real results and predictions (For binary results Loss is 'binary_crossentropy' and for categorical(more than one results) its 'categorical_crossentropy')
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the ANN on the training set
epochs_hist = ann.fit(X_train, y_train, batch_size = 25, epochs = 100, verbose = 1, validation_split = 0.2)
#validation_split is used for avoid overfitting and we are again dividing the data to generalise dataset for machine
#change the batch size and epochs to get better results

ann.summary()


#Evaluating the Model
epochs_hist.history.keys()


plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])



#Predicting any normal value
#0	46.58474463	58139.2591	3440.8237990000002	630059.0274	48013.6141
#1	33.48313022	39627.1248	9371.511070999999	319837.6593	17584.56963


X_test = np.array([[0, 46, 58000, 3500, 630000]])
X_Amount = ann.predict(X_test)
print('Expected Purchase Amount', X_Amount)




#Predicting the Test Set Results
y_pred = ann.predict(X_test)
#y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)),1))

#Making the COnfusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
