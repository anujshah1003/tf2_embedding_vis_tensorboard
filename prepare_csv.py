import os,cv2
import numpy as np
import pandas as pd

data_path = 'data'
data_dir_list = os.listdir(data_path)

class_to_labels={'cats':0,'dogs':1,'horses':2,'humans':3}

df = pd.DataFrame(columns=['img_names', 'labels', 'class_names'])

for dataset in data_dir_list:
    img_list=os.listdir(os.path.join(data_path, dataset))
    print ('Loading dataset-'+'{}\n'.format(dataset))
    label=class_to_labels[dataset]
    for img in img_list:
        img_name=os.path.join(data_path,dataset,img)
       # annotations.append([image_name,dataset,label])
        df = df.append({'img_names': img_name, 'labels': label,'class_names': dataset},ignore_index=True)

df.to_csv(os.path.join('data_annotations.csv'))
