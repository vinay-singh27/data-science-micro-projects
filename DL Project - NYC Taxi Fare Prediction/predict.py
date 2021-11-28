import torch
import pickle

from model import FarePredictionModel
from feature_engineering import *
from config import *

#load model
trained_model = FarePredictionModel(embedding_size= [(24, 12), (2, 1), (7, 4), (30, 15), (30, 15)],
                                num_of_numeric_feats= 4,
                                output_size = 1, layers = [200,100], p=0.4 )

trained_model.load_state_dict(torch.load(TRAINEDMODELDICT))
trained_model.eval()
print('Model is loaded')

#load cluster model
kmeans_model = pickle.load(open(CLUSTERMODEL, 'rb'))
print('kMeans model is loaded')


def test_data(dl_mdl, cluster_model): # pass in the name of the new model

    # INPUT NEW DATA
    plat = float(input('What is the pickup latitude?  '))
    plong = float(input('What is the pickup longitude? '))
    dlat = float(input('What is the dropoff latitude?  '))
    dlong = float(input('What is the dropoff longitude? '))
    psngr = int(input('How many passengers? '))
    dt = input('What is the pickup date and time?\nFormat as YYYY-MM-DD HH:MM:SS     ')
    
    # PREPROCESS THE DATA
    dfx_dict = {'pickup_latitude':plat,'pickup_longitude':plong,'dropoff_latitude':dlat,
         'dropoff_longitude':dlong,'passenger_count':psngr,'EDTdate':dt}
    dfx = pd.DataFrame(dfx_dict, index=[0])
    
    dfx = distance_features(dfx)  #distance features
    dfx = time_features(dfx)  #adding time features
    dfx = cluster_regions(dfx, num_of_cluster= 30, predict = True, model = kmeans_model) #clusters the regions
    
    # CREATE CAT AND CONT TENSORS
    categorical_cols = ['Hour', 'AMorPM', 'Weekday', 'pickup_cluster', 'dropoff_cluster']
    numerical_cols = ['passenger_count', 'dist_haversine', 'dist_dummy_manhattan', 'direction']
    
    #setting categorical cols
    for col in categorical_cols :
        dfx[col] = dfx[col].astype('category')

    #stacking categorical data
    xcats = np.stack([dfx[col].cat.codes.values for col in categorical_cols], axis=1)
    xcats = torch.tensor(xcats, dtype = torch.int64)

    xconts = np.stack([dfx[col].values for col in numerical_cols], 1)
    xconts = torch.tensor(xconts, dtype=torch.float)
    
    # PASS NEW DATA THROUGH THE MODEL WITHOUT PERFORMING A BACKPROP
    with torch.no_grad():
        z = trained_model(xcats, xconts)
    print(f'\nThe predicted fare amount is ${z.item():.2f}')


test_data(trained_model, kmeans_model)