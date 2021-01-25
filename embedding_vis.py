
import os,cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
#import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
#from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector

tf.__version__

LOG_DIR = 'embedding_logs_3'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
#%%
# prepare meta data file
data = pd.read_csv('data_annotations.csv',usecols=['img_names', 'labels', 'class_names'])

metadata_file = open(os.path.join(LOG_DIR, 'metadata_4_classes.tsv'), 'w')
metadata_file.write('Class\tName\n')

for label,name in zip(data.labels,data.class_names):
    metadata_file.write('{}\t{}\n'.format(name,label))
metadata_file.close()

#%%
#load fetures
with open('feature_vectors_400_samples.pkl', 'rb') as f:
    feature_vectors = pickle.load(f)
#feature_vectors = np.loadtxt('feature_vectors_400_samples.txt')
print ("feature_vectors_shape:",feature_vectors.shape)
print ("num of images:",feature_vectors.shape[0])
print ("size of individual feature vector:",feature_vectors.shape[1])

#%%   
#prepare sprite images         

img_data=[]
for img in tqdm(data.img_names):
    input_img=cv2.imread(img)
    input_img_resize=cv2.resize(input_img,(224,224))
    img_data.append(input_img_resize) 
img_data = np.array(img_data)

#%%
# Taken from: https://github.com/tensorflow/tensorflow/issues/6322
def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[...,np.newaxis], (1,1,1,3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
    # Inverting the colors seems to look better for MNIST
    #data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
            (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
            constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
            + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data
#%%
sprite = images_to_sprite(img_data)
cv2.imwrite(os.path.join(LOG_DIR, 'sprite_4_classes.png'), sprite)
#scipy.misc.imsave(os.path.join(LOG_DIR, 'sprite.png'), sprite)

#%%
features = tf.Variable(feature_vectors, name='features')
# Create a checkpoint from embedding, the filename and key are
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=features)
checkpoint.save(os.path.join(LOG_DIR, "embedding.ckpt"))

# Set up config
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    # Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path =  'metadata_4_classes.tsv'
    # Comment out if you don't want sprites
embedding.sprite.image_path =  'sprite_4_classes.png'
embedding.sprite.single_image_dim.extend([img_data.shape[1], img_data.shape[1]])
    # Saves a config file that TensorBoard will read during startup.

projector.visualize_embeddings(LOG_DIR, config)