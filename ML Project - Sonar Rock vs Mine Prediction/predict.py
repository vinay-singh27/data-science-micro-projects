import numpy as np
import pickle
from config import *

#load the model
trained_model = pickle.load(open(SAVEDMODEL, 'rb'))

# prediction = trained_model.predict(input_data)

# if (prediction[0]=='R'):
#   print('The object is a Rock')
# else:
#   print('The object is a mine')