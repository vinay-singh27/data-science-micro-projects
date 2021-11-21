#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics

from config import *

diabetes_data = pd.read_csv(TRAINDATA)

#feature engineering
diabetes_data['Age_Bucket'] = pd.cut(diabetes_data['Age'], bins= 4) #will create buckets for age
df_age = pd.get_dummies(diabetes_data['Age_Bucket'], drop_first=True)
diabetes_data = pd.concat([diabetes_data, df_age],axis=1)
diabetes_data.drop(['Age', 'Age_Bucket'], axis=1, inplace= True)


#separating data and labels
X = diabetes_data.drop(columns='Outcome', axis=1)
y = diabetes_data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, stratify=y, random_state=1)

#scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns= X_train.columns) #fit & transform the scaler on train data
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns= X_test.columns) #only transform on test data

#modeling
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, y_train)

#accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = metrics.accuracy_score(y_train, X_train_prediction) 
print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = metrics.accuracy_score(y_test, X_test_prediction) 
print('Accuracy on test data : ', test_data_accuracy)

#checking the classification report
print(metrics.classification_report(y_test, X_test_prediction))

#save the model
pickle.dump(classifier, open(SAVEDMODEL, 'wb'))

