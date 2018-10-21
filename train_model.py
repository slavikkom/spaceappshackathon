import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
import numpy as np
import time
from scipy.misc import imread, imresize
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import os


def create_model(number_of_feat=1, height=100, width=100):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3,3), input_shape=(number_of_feat * height, width, 4), padding='same')) 
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same'))  
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=(3,3), padding='same'))  
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same')) 
    model.add(LeakyReLU(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(LeakyReLU(0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("relu"))
    return model
    

def get_dependent_variable(cities=['Tallinn'], sources=['bt', 'city_lights', 'esm',\
                                                        'ndvi', 'population_count',\
                                                       'population_density', 'soil_sealing', 'srtm', 'vk_heatmap'],
                            img_height=100, img_width=100, years=[k for k in range(2000, 2018)]):
    X = np.zeros((len(cities) * len(years), len(sources), img_height, img_width, 4))
    last_valid = None
    for i in range(len(cities)):
        city = cities[i]
        for k in range(len(years)):
            year = years[k]
            for j in range(len(sources)):
                source = sources[j]
                try:
                    file = imread("data/" + city + "/" + city + "_" + source + "_" + str(year) + ".png")
                    img_array = np.array(file)
                    img_array = imresize(img_array, (img_height, img_width))
                    X[i*len(years) + k, j, :, :, :] = img_array
                    last_valid = img_array
                except:
                    if last_valid is not None:
                        X[i*len(years) + k, j, :, :, :] = last_valid
                    else:
                        X[i*len(years) + k, j, :, :, :] = np.zeros((img_height, img_width, 4))
    #X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2], X.shape[3], X.shape[4]))
    return X

def get_independent_variable(cities=['Tallinn'], years=[1989]):
    data = pd.read_csv("data\population_statistic\population.csv",encoding='latin1', sep="\t", header=None)
    data.columns = ["year","city/county/municipality","Population number", "Age unknown",\
                    "Population aged 0-14", "Population aged 15-64", "Population aged 65+", "Dependancy ratio"]
    y = []
    for city in cities:
        for i in range(len(data.index)):
            if city in data.loc[data.index[i], 'city/county/municipality'] and\
            int(data.loc[data.index[i], 'year']) in years:
                y.append(int(data.loc[data.index[i], "Population number"]))
    return np.array(y)
    
def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    image_array += np.random.randint(-10, 10)
    image_array[image_array < 0] = 0
    image_array[image_array > 255] = 255
    return image_array

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

def augement_data(X, y, n):
    y_mean = np.mean(y)
    y_std = np.std(y)
    augemented_X = np.zeros((n,X.shape[2], X.shape[3], X.shape[4], X.shape[5]))
    #np.zeros((n, X.shape[1], X.shape[2], X.shape[3], X.shape[4], X.shape[5]))
    augemented_y = np.zeros(n)
    for i in range(n):
        ii = np.random.randint(X.shape[0])
        jj = np.random.randint(X.shape[1])
        kk = np.random.randint(X.shape[2])
        
        #for j in range(X.shape[1]):
        for k in range(X.shape[2]):
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](X[ii, jj, kk, :, :, :])
            if ndarray.max(transformed_image) > 255 or ndarray.max(transformed_image) < 2:
                print(transformed_image)
                print(key)

            augemented_X[i, k, :, :, :] = transformed_image
        augemented_y[i] = y[ii] + np.random.randint(-1 * int(0.1 * y_std), int(0.1 * y_std))
        
    return augemented_X, augemented_y
    
    
def go():
    sources=['bt', 'city_lights', 'esm', 'ndvi', 'population_count',\
                                                           'population_density', 'soil_sealing', 'srtm', 'vk_heatmap']
    years=[k for k in range(2000, 2018)]
    X = get_dependent_variable(years=years)
    y = get_independent_variable(years=years)
    X1, y1 = augement_data(np.reshape(X, (1, X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4])), y, 10)
    train_X = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2], X.shape[3], X.shape[4]))
    train_y = y
    val_X = np.reshape(X1, (X1.shape[0], X1.shape[1] * X1.shape[2], X1.shape[3], X1.shape[4]))
    val_y = y1
    
    INIT_LR = 1e-3
    BATCH_SIZE = 32
    EPOCHS = 2

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=2, write_graph=True, write_images=True,
                                            write_grads=True)
    model = create_model(number_of_feat=len(sources))
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.adamax(lr=INIT_LR), 
        metrics=['mse']
    )
    model.fit(val_X, val_y, batch_size=BATCH_SIZE,
        epochs=EPOCHS, callbacks=[tbCallBack], validation_data=(train_X, train_y))
    if not os.path.exists("trained_models/"):
        os.makedir("trained_models/")
    model_path = "trained_models/model.hdf5"
    print("Saving model weights to " + model_path)
    model.save_weights(model_path)
    
    
    
if __name__ == '__main__':
    go()