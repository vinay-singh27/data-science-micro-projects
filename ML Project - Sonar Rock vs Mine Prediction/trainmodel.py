import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle 
from config import *

#load data
data = pd.read_csv(TRAINDATA, header= None)

#coverting label column into numeric
data[LABEL] = np.where(data[LABEL] == 'M', 1,0)

# separating data and Labels
X = data.drop(columns=LABEL, axis=1)
y = data[LABEL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, stratify=y, random_state = RANDOM_STATE)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#scaling
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns= X_train.columns) #fit & transform the scaler on train data
X_test = pd.DataFrame(scaler.fit_transform(X_test), columns= X_test.columns) #only transform on test data

#training the Logistic Regression model with training data
model = LogisticRegression()
model.fit(X_train, y_train)

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = metrics.accuracy_score(X_train_prediction, y_train) 
print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = metrics.accuracy_score(X_test_prediction, y_test) 
print('Accuracy on test data : ', test_data_accuracy)

#save the model
pickle.dump(model, open(SAVEDMODEL, 'wb'))



