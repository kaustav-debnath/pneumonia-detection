from __future__ import print_function

import pandas as pd
import numpy as np
import os
import sys
import traceback

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import resize
from sklearn.model_selection import train_test_split

import pickle
from tqdm import tqdm
import pydicom
import cv2
import glob, pylab

import PIL

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.models import Model,load_model

# from tensorflow.keras.applications.mobilenet import preprocess_input
# from sklearn.model_selection import GridSearchCV


prefix = '/opt/program'
model_prefix = '/opt/ml/model'
# prefix = './'

IMG_ACTUAL_SIZE = 1024
input_path = os.path.join(prefix , 'input/data/')
output_path = os.path.join(prefix, 'output/')
model_path = os.path.join(prefix, 'model/')

training_path = os.path.join(input_path, 'stage_2_train_images/')
training_labels_path = input_path + 'stage_2_train_labels.csv'

print(training_path)
print(training_labels_path)

def create_model(X_train, y_train, model_name):
    #classifier = load_model(os.path.join(model_path,'mobilenet_model.h5'))
    print('model name=>',model_name)
    model_path_storage = ''
    if model_name=='UNet':
        classifier = load_model(os.path.join(model_path,'unet_model.h5'))
        model_path_storage = os.path.join(model_prefix,'unet-prediction.h5')
    else:
        return 'Please provide a valid model to train'
        
    # Compiling the Loaded Model
    if model_name=='UNet':
        classifier.compile(
            optimizer='Adam',
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]
        )
    else:
        return 'Please provide a valid model to compile'

    # declare the callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.0001, verbose=1),
        ModelCheckpoint(model_path_storage, verbose=1, save_best_only=True, save_weights_only=True)
    ]
    # Fit gridsearch to training set
    history = classifier.fit(
        X_train,
        y_train,
        batch_size=32, 
        epochs=20, 
        callbacks=callbacks,
        validation_split = 0.2
    )
    return classifier, history


# method to create masks for UNet bounding box prediction
def create_masks(data,img_width,img_height,channels,image_count):
    X_train_mask = np.zeros((image_count,img_width,img_height,3), dtype=np.float32)
    # y_train_mask = np.zeros((image_count,2), dtype=np.float32)
    index_count = 0
    img_original_size = 1024
    factor = img_original_size / img_width
    masks = np.zeros((image_count, img_width, img_height, channels))
    for key in tqdm(data):
        # mask = np.zeros((1024, 1024), dtype=np.uint8)
        if data[key]['target'] == 1:
            ds = pydicom.dcmread(data[key]['image_path'])
            res = load_image(data[key]['image_path'],img_width,img_height)
            res = resize(res, (img_width, img_height), mode='edge', preserve_range=True)
            # res = np.stack((res,) * 3, -1)
            for i in range(len(data[key]['bboxes'])):
                # mask = ds.pixel_array
                x1 = int(data[key]['bboxes'][i][0])
                y1 = int(data[key]['bboxes'][i][1])
                x2 = int(x1 + data[key]['bboxes'][i][2])
                y2 = int(y1 + data[key]['bboxes'][i][3])
                # mask = cv2.rectangle(mask, (x1,y1), (x2,y2), (0,255,0), -1)
                # mask = resize(mask, (img_width, img_height, channels), mode = 'edge', preserve_range = True)
                # mask = mask/255.0
                x1_mask = int(x1 / factor)
                y1_mask = int(y1 / factor)
                x2_mask = int(x2 / factor)
                y2_mask = int(y2 / factor)
                masks[index_count][y1_mask:y2_mask, x1_mask:x2_mask] = 1
                
            X_train_mask[index_count] = res
            # y_train_mask[index_count] = mask
            index_count+=1
        else:
            continue
        if index_count>=image_count:
            break
    return X_train_mask, masks


# method to load the image and read .dcm file using pydicom
def load_image(image_path,img_width,img_height):
    #this method helps in reading .dcm file
    ds = pydicom.dcmread(image_path)
    img = ds.pixel_array
    #calling cv2.resize to reduce the dimension of image
    res = cv2.resize(img, (img_width,img_height), interpolation=cv2.INTER_AREA)
    #converting the image array data into float 
    res = res.astype(np.float32)/255
    cv2.destroyAllWindows()
    res = np.stack((res,) * 3, -1)
    return res



# create a dictionary of metadata info from training labels file for UNet
def create_dictionary_image_detection(data):
    metainfo = {}
    for index, row in data.iterrows(): 
        patientid = row['patientId']
        if row['Target'] == 1:
            if patientid not in metainfo:
                metainfo[patientid] = {
                    'target': row['Target'],
                    'bboxes': [],
                    'image_path': os.path.join(training_path,patientid + '.dcm')
                }
            metainfo[patientid]['bboxes'].append([row['x'],row['y'],row['width'],row['height']])
    return metainfo


def train():
    print('Starting the training.')
    model_name = ''
    try:
        df_train_labels = pd.read_csv(training_labels_path)
        df_train_labels = df_train_labels.fillna(0) #data imputation replacing NAN values with 0s
        metainfo = create_dictionary_image_detection(df_train_labels)
        
        print('Starting Training and Printing Unet bounding box prediction*************************************************************')
        X, y = create_masks(metainfo, 128,128,3,6012)
        classifier,history = create_model(X, y, 'UNet')
        print('Starting Training and Printing Unet bounding box prediction*************************************************************')
        print('Training is complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failure
        # Reason in the DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs
        print(
            'Exception during training: ' + str(e) + '\n' + trc,
            file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
    return classifier,history
            
if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
    