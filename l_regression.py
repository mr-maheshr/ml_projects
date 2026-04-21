import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load data
sonar_data = pd.read_csv('sonar data.csv', header=None)

#View first 5 raws
sonar_data.head()

#No of raws & columns
sonar_data.shape

# Statistical measures of the data
sonar_data.describe()

#counts how many times each unique value appears in label column
sonar_data[60].value_counts()

#mean of each numeric column
sonar_data.groupby(60).mean()

#if needs multiple stats
#sonar_data.groupby(60).agg(['mean', 'std', 'min', 'max'])

# Separate data & label
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

#Train & test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)

#LogisticRegression
model = LogisticRegression()

#Train model
model.fit(X_train, Y_train)

#Accuracy -> training data
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data: ', train_data_accuracy)

#Accuracy -> test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on test data: ', test_data_accuracy)

#Making Predictive System
#Rock instance
#input_data = (0.0079,0.0086,0.0055,0.0250,0.0344,0.0546,0.0528,0.0958,0.1009,0.1240,0.1097,0.1215,0.1874,0.3383,0.3227,0.2723,0.3943,0.6432,0.7271,0.8673,0.9674,0.9847,0.9480,0.8036,0.6833,0.5136,0.3090,0.0832,0.4019,0.2344,0.1905,0.1235,0.1717,0.2351,0.2489,0.3649,0.3382,0.1589,0.0989,0.1089,0.1043,0.0839,0.1391,0.0819,0.0678,0.0663,0.1202,0.0692,0.0152,0.0266,0.0174,0.0176,0.0127,0.0088,0.0098,0.0019,0.0059,0.0058,0.0059,0.0032)

#Mine instance
input_data = (0.0134,0.0172,0.0178,0.0363,0.0444,0.0744,0.0800,0.0456,0.0368,0.1250,0.2405,0.2325,0.2523,0.1472,0.0669,0.1100,0.2353,0.3282,0.4416,0.5167,0.6508,0.7793,0.7978,0.7786,0.8587,0.9321,0.9454,0.8645,0.7220,0.4850,0.1357,0.2951,0.4715,0.6036,0.8083,0.9870,0.8800,0.6411,0.4276,0.2702,0.2642,0.3342,0.4335,0.4542,0.3960,0.2525,0.1084,0.0372,0.0286,0.0099,0.0046,0.0094,0.0048,0.0047,0.0016,0.0008,0.0042,0.0024,0.0027,0.0041)

# Changing input data to numpy array
input_data_array = np.asarray(input_data)

# Reshape the np array -> as we predict for one instance
input_data_reshaped = input_data_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')