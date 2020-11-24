#Project 7 on Yelp Reviews Using NLP

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#Importing the Dataset
yelp_df = pd.read_csv('yelp.csv')

#It will give all the info about datatype and null objects
yelp_df.info()

#It gives the max and low and some mean values of the whole data
yelp_df.describe()

yelp_df.text[0:2]
yelp_df['text'][0]


#Visualize the Data
#Creating the new column Length to get the length of the text
yelp_df['length'] = yelp_df['text'].apply(len)

#Plotting a histogram with 100 bins for length column
yelp_df['length'].plot(bins = 100, kind = 'hist')

yelp_df.length.describe()

#Printing the biggest message
yelp_df[yelp_df['length'] == 4997]['text'].iloc[0]
yelp_df[yelp_df['length'] == 1]['text'].iloc[0]
yelp_df[yelp_df['length'] == 710]['text'].iloc[0]

#Counting the Reviews
sns.countplot(y = 'stars', data = yelp_df)

#PLotting the length for different star reviews
g = sns.FacetGrid(data = yelp_df, col = 'stars', col_wrap = 3)
g.map(plt.hist, 'length', bins = 20, color = 'g')

#Merging the data for 1 Star and 5 Star Review
yelp_df_1 = yelp_df[yelp_df['stars'] == 1]
yelp_df_5 = yelp_df[yelp_df['stars'] == 5]
yelp_df_1
yelp_df_5

yelp_df_1_5 = pd.concat([yelp_df_1, yelp_df_5])
yelp_df_1_5

#Show the percentage of 1 Star and 5 Star Reviews
print('One Star Percentage Review = ' , (len(yelp_df_1)/len(yelp_df_1_5))*100, '%')
print('Five Star Percentage Review = ' , (len(yelp_df_5)/len(yelp_df_1_5))*100, '%')

sns.countplot(yelp_df_1_5['stars'])


#Mini Challenge
mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'

import string
string.punctuation
from nltk.corpus import stopwords
stopwords.words('English')

mc_clean = [char  for char in mini_challenge if char not in string.punctuation ]
mc_clean_join = ''.join(mc_clean)
mc_clean_join

mc_final = [word   for word in mc_clean_join.split() if word.lower() not in stopwords.words('English')]
mc_final


#COunt Vectorizer
sample_data = ['This document is the first document.', 'This document is the second document.', 'Is this the first document!' ]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

cv = vectorizer.fit_transform(sample_data) 
vectorizer.get_feature_names()
cv.toarray()


#Cleaning the Texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # It just takes the root of the words(words which make sense)
corpus = []
for i in range(0, 10000):
    review = re.sub('[^a-zA-z]', ' ', yelp_df['text'][i]) #------Replace all the punctuations apart from A-z to spaces
    review = review.lower() #-------------Make all the letters to lower case
    review = review.split() #-------------- Split all the words
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    #all_stopwords.remove('not') # as not was included in stopwrds, so we are removing it from there as it is essential work to predict the results
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)] #------- Single row for loop
    review = ' '.join(review)
    corpus.append(review)
    
print(corpus)


#Create the Bag Words of Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() #----- to remove some unique words whihc are not frequent like names which will not help to rpedict the review
X = cv.fit_transform(corpus).toarray() # fit_transform put the words into columns
print(cv.get_feature_names())
y = yelp_df['stars']

len(X[0])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Training the Naive Bayes Model on the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


#testing_sample = ['Amazing Food! highly recommendable.']
testing_sample = ['shit food, made me sick!']

test_vec = cv.transform(testing_sample)
test_predict = classifier.predict(test_vec)
test_predict



