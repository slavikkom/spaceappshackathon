# spaceappshackathon
Main functionality is located in console.py. This file takes argument --city, which specifies folder where to look for input data. Input data should be images with brightness temperature, Normalized Difference Vegetation Index (NDVI), soil sealing percentage, NASA night lights product, Terrain Ruggeddness Index, SRTM digital elevation model, VK.com geotagged photographs, road network data, population and population density data for specific region.
As output will be provided estimated number of population in given region. By default, it uploads model weights from trained_models/model.hdf5 file.
To train new model, run train_model.py. By default it uses samples from data/Tallinn folder, and creates 10 augemented images for each map. Model will be saved to train_models folder. Folder augemented_examples contains examples of original and augemented images. Original image will always be with index 0.
In data/population_statistic folder there is a file with number of people in each Estonian city for last years. Population number column was used as target variable. 
Folder data/Tallinn contains example of input data. 
