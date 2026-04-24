
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

news_data = pd.read_csv('news.csv')

# number of raws and columns
news_data.shape

# first 5 raws
news_data.head()

# null values
news_data.isnull().sum()

# replace null values
news_data = news_data.fillna(' ')

# merge title and author
news_data['content'] = news_data['author']+' '+news_data['title']

print(news_data['content'])

# data split
X = news_data.drop(columns='label', axis=1)

# map 'fake' to 0 and 'real' to 1
news_data['label'] = news_data['label'].map({'fake': 0, 'real': 1})
Y = news_data['label']

print(X)
print(Y)

# stem -> reduce a word to its root word
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_data['content'] = news_data['content'].apply(stemming)

print(news_data['content'])

# feature and label
X = news_data['content'].values
Y = news_data['label'].values

print(X)

print(Y)

# convert -> text to numeric
vect = TfidfVectorizer()
vect.fit(X)

X = vect.transform(X)

print(X)

# train test split
# Identify and remove rows where Y is NaN
not_nan_indices = ~np.isnan(Y)
X_filtered = X[not_nan_indices]
Y_filtered = Y[not_nan_indices]

X_train, X_test, Y_train, Y_test = train_test_split(X_filtered, Y_filtered, test_size = 0.2, stratify=Y_filtered, random_state=2)

# train Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

# accuracy -> training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy -> test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

# predictive system
X_new = X_test[0]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Fake')
else:
  print('The news is Real')

print(Y_test[0])
