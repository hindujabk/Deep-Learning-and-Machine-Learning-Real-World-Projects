#Project 6 - NLP on EMails SPam

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('emails.csv')
dataset.shape

#Cleaning the Texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # It just takes the root of the words(words which make sense)
corpus = []
for i in range(0, 5728):
    review = re.sub('[^a-zA-z]', ' ', dataset['text'][i]) #------Replace all the punctuations apart from A-z to spaces
    review = review.lower() #-------------Make all the letters to lower case
    review = review.split() #-------------- Split all the words
    ps = PorterStemmer()
#    all_stopwords = stopwords.words('english')
 #   all_stopwords.remove('not') # as not was included in stopwrds, so we are removing it from there as it is essential work to predict the results
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #------- Single row for loop
    review = ' '.join(review)
    corpus.append(review)
    
print(corpus)


#Create the Bag Words of Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #----- to remove some unique words whihc are not frequent like names which will not help to rpedict the review
X = cv.fit_transform(corpus).toarray() # fit_transform put the words into columns
y = dataset.iloc[:, -1].values

len(X[0])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Naive Bayes Model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)