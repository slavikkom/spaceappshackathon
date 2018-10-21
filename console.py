import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
import numpy as np
import time
from os import system
import sys
import argparse
from train_model import *
    
    
def go(city):
    year = 2005
    sources=['bt', 'city_lights', 'esm',\
                                                            'ndvi', 'population_count',\
                                                           'population_density', 'soil_sealing', 'srtm', 'vk_heatmap']
    model = create_model(number_of_feat=len(sources))
    model.load_weights("trained_models/model.hdf5")
    print("Architecture used:\n")
    time.sleep(1)
    print(model.summary())
    time.sleep(1)
    system('cls')
    print("Prediction in progress for " + city + ":")
    for i in range(101):
        time.sleep(0.01)
        sys.stdout.write("\r%d%%" % i)
        sys.stdout.flush()
    print()
    X = get_dependent_variable(cities=[city], sources=sources,
                                img_height=100, img_width=100, years=[year])
    X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2], X.shape[3], X.shape[4]))
    print("Model prediction (population estimation) for city " + city + " is: " + str(int(model.predict(X)[0][0])) + " people")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify path to data')
    parser.add_argument('--city',
                        help='path to city files')
    parameters = parser.parse_args()
    go(parameters.city)
       