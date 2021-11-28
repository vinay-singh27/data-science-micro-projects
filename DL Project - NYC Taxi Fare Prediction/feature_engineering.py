import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import pickle

from distance_functions import *
from config import *

def distance_features(df) :

    '''
    Calcualte the distance features using the provided latitude & longitude
    
    '''

    #haversine df
    df['dist_haversine'] = haversine_distance(df['pickup_latitude'].values, 
                                                df['pickup_longitude'].values, df['dropoff_latitude'].values, 
                                                df['dropoff_longitude'].values)

    #manhattan df
    df['dist_dummy_manhattan'] = dummy_manhattan_distance(df['pickup_latitude'].values, 
                                                            df['pickup_longitude'].values, 
                                                            df['dropoff_latitude'].values, 
                                                            df['dropoff_longitude'].values)

    #direction of travel
    df['direction'] = bearing_array(df['pickup_latitude'].values, df['pickup_longitude'].values, 
                                    df['dropoff_latitude'].values, df['dropoff_longitude'].values)

    return df



def time_features(df) :

    df['Daylight_datetime'] = pd.to_datetime(df['pickup_datetime'].str[:19]) - pd.Timedelta(hours=4) #due to daylight savings 
    df['Hour'] = df['Daylight_datetime'].dt.hour  #time of pickup
    df['AMorPM'] = np.where(df['Hour']<12,'am','pm')
    df['Weekday'] = df['Daylight_datetime'].dt.strftime("%a")

    return df


def cluster_regions(df, num_of_cluster, batch_size=10000,  predict = False, trained_model = None) :

    if predict == False :
        #stack latitude & longitude df
        coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values,
                        df[['dropoff_latitude', 'dropoff_longitude']].values))

        #using 20 clusters
        fit_model = MiniBatchKMeans(n_clusters= num_of_cluster, batch_size=batch_size)
        fit_model.fit(coords)

        #save the kmeans model
        pickle.dump(fit_model, open(CLUSTERMODEL, 'wb'))

    else : 
        fit_model = trained_model

    df['pickup_cluster'] = fit_model.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = fit_model.predict(df[['dropoff_latitude', 'dropoff_longitude']])

    return df


        

