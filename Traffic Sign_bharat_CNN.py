#Project 5 - Traffic Signal Signs for Automated Driving
'''In this case study, you have been provided with images of traffic signs and the goal is to train a Deep Network to classify them

The dataset contains 43 different classes of images.

Classes are as listed below:

( 0, b'Speed limit (20km/h)') ( 1, b'Speed limit (30km/h)')
( 2, b'Speed limit (50km/h)') ( 3, b'Speed limit (60km/h)')
( 4, b'Speed limit (70km/h)') ( 5, b'Speed limit (80km/h)')
( 6, b'End of speed limit (80km/h)') ( 7, b'Speed limit (100km/h)')
( 8, b'Speed limit (120km/h)') ( 9, b'No passing')
(10, b'No passing for vehicles over 3.5 metric tons')
(11, b'Right-of-way at the next intersection') (12, b'Priority road')
(13, b'Yield') (14, b'Stop') (15, b'No vehicles')
(16, b'Vehicles over 3.5 metric tons prohibited') (17, b'No entry')
(18, b'General caution') (19, b'Dangerous curve to the left')
(20, b'Dangerous curve to the right') (21, b'Double curve')
(22, b'Bumpy road') (23, b'Slippery road')
(24, b'Road narrows on the right') (25, b'Road work')
(26, b'Traffic signals') (27, b'Pedestrians') (28, b'Children crossing')
(29, b'Bicycles crossing') (30, b'Beware of ice/snow')
(31, b'Wild animals crossing')
(32, b'End of all speed and passing limits') (33, b'Turn right ahead')
(34, b'Turn left ahead') (35, b'Ahead only') (36, b'Go straight or right')
(37, b'Go straight or left') (38, b'Keep right') (39, b'Keep left')
(40, b'Roundabout mandatory') (41, b'End of no passing')
(42, b'End of no passing by vehicles over 3.5 metric tons')
The network used is called Le-Net that was presented by Yann LeCun http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
'''


# import libraries 
import pickle
import seaborn as sns
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random

# The pickle module implements binary protocols for serializing and de-serializing a Python object structure.
with open("./traffic-signs-data/train.p", mode='rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode='rb') as validation_data:
    valid = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode='rb') as testing_data:
    test = pickle.load(testing_data)
    
    
X_train, y_train = train['features'], train['labels']
X_train.shape
X_test, y_test = test['features'], test['labels']
X_valid, y_valid = valid['features'], valid['labels']

X_test.shape


#Image Exploration
i = 1000
plt.imshow(X_train[i])
y_train[i]


#Data Preparation
#from sklearn.utils import shuffle
#X_train, y_train = shuffle(X_train, y_train)

#Converting the Images into gray scale
X_train_gray = np.sum(X_train/3, axis = 3, keepdims = True)
X_train_gray.shape
X_test_gray  = np.sum(X_test/3, axis=3, keepdims=True)
X_valid_gray  = np.sum(X_valid/3, axis=3, keepdims=True) 

#Normalising the Data(Feature Sclaing)
X_train_gray_norm = (X_train_gray - 128)/128
X_test_gray_norm = (X_test_gray - 128)/128
X_valid_gray_norm = (X_valid_gray - 128)/128
X_train_gray_norm #all images will be from -1 to 1


#Image Exploration
i = 1000
plt.imshow(X_train_gray[i].squeeze(), cmap = 'gray')

i = 1000
plt.imshow(X_train_gray_norm[i].squeeze(), cmap = 'gray')


import tensorflow as tf
#Building the CNN

#Initialising the CNN
cnn = tf.keras.models.Sequential()

#Step1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters = 6, kernel_size = (5,5), activation = 'relu', input_shape = [32,32,1]))

#Step2 - Pooling
cnn.add(tf.keras.layers.AveragePooling2D())

#Adding a Second Concolutional Layer    
cnn.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu'))#, input_shape = [64,64,3])) ---- As this input parameter will hv to apply only in first layer as it is connected with input
cnn.add(tf.keras.layers.AveragePooling2D())       

#Step3 - Flattening
cnn.add(tf.keras.layers.Flatten())

#Step4 - Full Connection
cnn.add(tf.keras.layers.Dense(units = 120, activation = 'relu'))#---Units is number of neurons to add
cnn.add(tf.keras.layers.Dense(units = 84, activation = 'relu'))


#Step5 - Output Layer
cnn.add(tf.keras.layers.Dense(units = 43, activation = 'softmax'))#---Units is 1 because we are doing binary classification


#Training the CNN

#Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'] )

#Training the CNN on training set and Eveluating it on the Test Set
history = cnn.fit(X_train_gray_norm, y_train, validation_data = (X_valid_gray_norm, y_valid), batch_size = 300, epochs = 25)


#Model Evaluation
score = cnn.evaluate(X_test_gray_norm, y_test)
print('Test Accuracy:' .format(score))

history.history.keys()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))
epochs

plt.plot(epochs, accuracy, 'bo', label = 'Training Accuracy') #bo is blue colour with dots
plt.plot(epochs, val_accuracy, 'b', label = 'Validation Accuracy')
plt.legend()


plt.plot(epochs, loss, 'ro', label = 'Training Loss') 
plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
plt.legend()


#Confusion Matrix
y_predict = cnn.predict_classes(X_test_gray_norm)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_predict, y_test)
cm
plt.figure(figsize = (25,25))
sns.heatmap(cm, annot = True)

#Plot the images with predicted and true labels

L = 7
W = 7
fig, axes = plt.subplots(L, W, figsize = (12,12))
axes = axes.ravel()#Flatening the array

for i in np.arange(0, L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('PC: {}\n Act: {}'.format(y_predict[i], y_true[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 1, hspace = 1)