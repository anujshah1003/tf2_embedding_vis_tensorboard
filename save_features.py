# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential

print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=True)

# creating new model by eliminating the last layer
cust_model = Sequential()
for layer in model.layers[:-1]: # just exclude last layer from copying
    cust_model.add(layer)

# read the annotation file
data = pd.read_csv('data_annotations.csv',usecols=['img_names', 'labels', 'class_names'])

# Feature extraction function
# (num_data,features_size)
def get_image_features(image_file_name):
    
    image = load_img(image_file_name, target_size=(224, 224))
    image = img_to_array(image)
	# preprocess the image by (1) expanding the dimensions and
	# (2) subtracting the mean RGB pixel intensity from the
	# ImageNet dataset
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    features = cust_model.predict(image)
    
    #for output after conv_layers
    #features = features.reshape((features.shape[0], features.shape[1]*features.shape[2]*features.shape[3]))

    return features

image_features_list=[]

for img in tqdm(data.img_names):
    image_features=get_image_features(img)
    image_features_list.append(image_features)    

image_features_arr=np.asarray(image_features_list)
del image_features_list # del to get free space
image_features_arr = np.rollaxis(image_features_arr,1,0)
image_features_arr = image_features_arr[0,:,:]
pickle.dump(image_features_arr, open('feature_vectors_400_samples.pkl', 'wb'))
#np.savetxt('feature_vectors_400_samples.txt',image_features_arr)
#feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
# (num_data,features_size)
