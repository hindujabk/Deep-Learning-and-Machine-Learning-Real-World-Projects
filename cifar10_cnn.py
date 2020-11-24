#Project 2 for CNN with 10 Different classes
"""
-AirPlanes
-Cars
-Birds
-Cats
-Deer
-Dogs
-Frogs
-Horses
-Ships
-Trucks
"""

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import tensorflow as tf

#Importing the Dataset
from keras.datasets import cifar10
 (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train.shape
Y_train.shape

X_test.shape
Y_test.shape

#Visualize the Data

#Visualize the single image
i = 10000
plt.imshow(X_train[i])#----X_train has a image
print(Y_train[i])#-----Y_train has assigned with a above class number mentioned

#Visuslaize the multiple images by creating grid
W_grid = 15
L_grid = 15

fig, axes = plt.subplots(L_grid, W_grid, figsize = (25,25)) # it will plot the 15*15 array which consists of 225 blank plots
axes = axes.ravel() #----Flatten the array whihc is 15*15 = 225

n_training = len(X_train) #putting all images in n_training
print(n_training)

#Lets fill images in these blank plots
for i in np.arange(0, W_grid * L_grid):
    index = np.random.randint(0, n_training) #it will pick a random number from 0 to 50000 which is n_training
    axes[i].imshow(X_train[index])
    axes[i].set_title(Y_train[index])
    axes[i].axis('off') #---It will make axis disable 
    
plt.subplots_adjust(hspace = 0.4)#----it will create the spaces between plots


#Data Preparation
X_train = X_train.astype('float32')#---making images into float
X_test = X_test.astype('float32')#---making images into float

number_cat = 10 #number of categories

Y_train #Y_train is a class label and its a decimal value and we hv to make it to Binary

import keras
Y_train = keras.utils.to_categorical(Y_train, number_cat)#we hv to mention number of categories to make it 0000000001 otherwise it will just take 01    

Y_train #now it will become binary


Y_test = keras.utils.to_categorical(Y_test, number_cat)
Y_test

#Performing data normalisation
X_train = X_train/255
X_test = X_test/255

X_train
X_train.shape

Input_shape = X_train.shape[1:]

Input_shape


#Training the Model

#Initialising the CNN
cnn = tf.keras.models.Sequential()

#Step1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = Input_shape))

#Step2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(2,2))
cnn.add(tf.keras.layers.Dropout(0.4))

#Adding a Second Concolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(2,2)) 
cnn.add(tf.keras.layers.Dropout(0.4))      
                
#Step3 - Flattening
cnn.add(tf.keras.layers.Flatten())

#Step4 - Full Connection
cnn.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))#---Units is number of neurons to add

cnn.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))#---Units is number of neurons to add

#relu function is something which is continuous
#Step5 - Output Layer
cnn.add(tf.keras.layers.Dense(units = 10, activation = 'softmax'))#---Units is 10 because we have 10 classes and 'softmax'



#Training the CNN

#Compiling the CNN
cnn.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

#Training the CNN on training set and Eveluating it on the Test Set
result = cnn.fit(X_train, Y_train, batch_size = 32, epochs = 2, shuffle = True)
#result = cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


#Evaluate the Model

evaluation = cnn.evaluate(X_test, Y_test)
print('Test Accuracy:'.format(evaluation[1]))

y_predict = cnn.predict_classes(X_test)
y_predict

Y_test # we are changing from binary to decimal now

Y_test = Y_test.argmax(1)
Y_test

#Visualise the Images in Subplots with actual and predicted results

L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes.ravel()

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction = {} \n True = {}'.format(y_predict[i], Y_test[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 1)

#Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(Y_test, y_predict)
cm
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot = 'True')


