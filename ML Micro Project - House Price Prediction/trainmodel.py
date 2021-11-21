#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

from config import *
from optuna_tuning import *

#load model
boston_dataset = datasets.load_boston()
house_price_df = pd.DataFrame(data= boston_dataset['data'], columns= boston_dataset['feature_names'])
house_price_df['price'] = boston_dataset['target']

# separating data and labels
# scaling is not required as we would be using tree based model
# we will also create validation dataset as we will be tuning the hyperparameters of the model
X = house_price_df.drop(columns='price', axis=1)
y = house_price_df['price']

X_train, X_add, y_train, y_add = train_test_split(X, y, test_size = 0.3,  random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_add, y_add, test_size = 0.5,  random_state=1)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

#hyperparameter tuning
optimized_model_param = hyperparameter_tuning(X_train, y_train, X_val, y_val)

#train on best features
optimize_model = LGBMRegressor(**optimized_model_param)
optimize_model.fit(X_train, y_train)

# prediction on training data
training_data_prediction = optimize_model.predict(X_train)
r2_score_train = metrics.r2_score(y_train, training_data_prediction)
mae_train = metrics.mean_absolute_error(y_train, training_data_prediction)
print("R squared error Train: ", r2_score_train)
print('Mean Absolute Error Train: ', mae_train)


# prediction on test data
test_data_prediction = optimize_model.predict(X_test)
r2_score_test = metrics.r2_score(y_test, test_data_prediction)
mae_test = metrics.mean_absolute_error(y_test, test_data_prediction)
print("R squared error Test: ", r2_score_test)
print('Mean Absolute Error Test: ', mae_test)


#save the model
import pickle 
pickle.dump(optimize_model, open(SAVEDMODEL, 'wb'))