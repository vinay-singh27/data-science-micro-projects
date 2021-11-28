#import libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *
from feature_engineering import *
from model import FarePredictionModel, train_test_split

#load data
data = pd.read_csv(TRAINDATA)
data.drop('fare_class', axis=1, inplace= True) #will drop the fare class column

'''
Feature Engineering:
Distance Features - Adding Haversine distance, Manhattan distance & bearing array(to calculate direction)
Time Features - Adding day & time features
Cluster Features - Creating clusters of latitude & longitude
'''
data = distance_features(data)  #distance features
data = time_features(data)  #adding time features
data = cluster_regions(data, num_of_cluster= 30, predict= False) #clusters the regions


#separating categorical & numeric features
categorical_cols = ['Hour', 'AMorPM', 'Weekday', 'pickup_cluster', 'dropoff_cluster']
numerical_cols = ['passenger_count', 'dist_haversine', 'dist_dummy_manhattan', 'direction']
y_col = ['fare_amount']

#setting categorical cols
for col in categorical_cols :
    data[col] = data[col].astype('category') 
#stacking categorical data
categorical_data = np.stack([data[col].cat.codes.values for col in categorical_cols], axis=1)
categorical_data = torch.tensor(categorical_data, dtype = torch.int64)
#creating categorical sizes & embedding size
cat_sizes = [len(data[col].cat.categories) for col in categorical_cols]
embedding_sizes = [(size, min(50, (size+1)//2)) for size in cat_sizes]

#stacking numerical data
numerical_data = np.stack([data[col].values for col in numerical_cols], axis=1)
numerical_data = torch.tensor(numerical_data, dtype = torch.float)

#create label
y = torch.tensor(data[y_col].values, dtype = torch.float).reshape(-1,1)


#modeling
torch.manual_seed(42)
model = FarePredictionModel(embedding_sizes, numerical_data.shape[1], 1, [200,100], p=0.4)

criterion = nn.MSELoss()  # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #optimizer

#train & test split
cat_train, cat_test, con_train, con_test, y_train, y_test = train_test_split(categorical_data,
                                                        numerical_data, y,
                                                        batch_size = 60000,
                                                        test_ratio = 0.2)


#train the model
import time
start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss)
    
    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed


# performance on test data
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))
print(f'RMSE: {loss:.8f}')


# save the pytorch model
print('SAVING THE MODEL')
torch.save(model.state_dict(), TRAINEDMODELDICT)
