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
    if model_name=='MobileNet':
        classifier = load_model(os.path.join(model_path,'mobilenet_model.h5'))
        model_path_storage = os.path.join(model_prefix,'mobilenet-classification.h5')
    elif model_name=='VGGNet':
        classifier = load_model(os.path.join(model_path,'base_model_vgg.h5'))
        model_path_storage = os.path.join(model_prefix,'vggnet-classification.h5')
    elif model_name=='InceptionV3':
        classifier = load_model(os.path.join(model_path,'base_model_Inc.h5'))
        model_path_storage = os.path.join(model_prefix,'inception-classification.h5')
        
    # Compiling the Loaded Model
    classifier.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # declare the callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(model_path_storage, verbose=1, save_best_only=True, save_weights_only=True)
    ]
    # Fit gridsearch to training set
    history = classifier.fit(
        X_train,
        y_train,
        batch_size=32, 
        epochs=1, 
        callbacks=callbacks,
        validation_split = 0.2
    )
    return classifier, history

# method to prepare training data and labels    
def prepare_data(data,img_width,img_height):
#     X_train = []
#     y_train = []
    X_train = np.zeros((len(data),img_width,img_height,3), dtype=np.float32)
    y_train = np.zeros((len(data),2), dtype=np.float32)
    index=0
    for key in data:
        img = load_image(data[key]['image_path'],img_width,img_height)
        label = data[key]['target']
        if label == 1:
            encoded_label = [1,0]
        else:
            encoded_label = [0,1]
#         X_train.append(img)
#         y_train.append(encoded_label)
        X_train[index] = img
        y_train[index] = encoded_label
        index += 1
    print('training data processed==>'+str(X_train.shape))
    print('validation data processed==>'+str(y_train.shape))
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     X_train = np.stack((X_train,) * 3, -1)
    return X_train, y_train

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

# draw an image with detected objects
def draw_image_with_boxes(res, boxes_list,factor):
#      print(boxes_list)
     # plot the image
     plt.imshow(res, cmap=plt.cm.bone)
     # get the context for drawing boxes
     ax = plt.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
#           x1, y1, width, height = box
            x1 = box[0]/factor
            y1 = box[1]/factor
            width = box[2]/factor
            height = box[3]/factor
            rect = Rectangle((x1,y1), width, height, ec='k', fc='none')
            ax.add_patch(rect)
     # show the plot
     plt.show()

# create a dictionary of metadata info from training labels file
def create_dictionary(data):
    metainfo = {}
    for index, row in data.iterrows(): 
        patientid = row['patientId']
        # print(patientid)
        if patientid not in metainfo:
            metainfo[patientid] = {
                'target': row['Target'],
                'bboxes': [],
                'image_path': os.path.join(training_path,patientid + '.dcm')
            }
        if metainfo[patientid]['target'] == 1:
            metainfo[patientid]['bboxes'].append([row['x'],row['y'],row['width'],row['height']])
    return metainfo

def train():
    print('Starting the training.')
    model_name = ''
    try:
        df_train_labels = pd.read_csv(training_labels_path)
        df_train_labels = df_train_labels.fillna(0) #data imputation replacing NAN values with 0s
        metainfo = create_dictionary(df_train_labels)
        X, y = prepare_data(metainfo,128,128)
#         print('Starting Training and Printing MobileNet classification*************************************************************')
#         classifier,history = create_model(X, y, 'MobileNet')
#         print('Ending Training and Printing  MobileNet classification*************************************************************')
        print('/******************************************************************************************************************/')
        print('/******************************************************************************************************************/')
        print('Starting Training and Printing VggNet classification*************************************************************')
        classifier,history = create_model(X, y, 'VGGNet')
        print('Ending Training and Printing  VggNet classification*************************************************************')
        print('/******************************************************************************************************************/')
        print('/******************************************************************************************************************/')
#         X_inception, y_inception = prepare_data(metainfo,299,299)
#         print('Starting Training and Printing Inception classification*************************************************************')
#         classifier,history = create_model(X_inception, y_inception, 'InceptionV3')
#         print('Ending Training and Printing  Inception classification*************************************************************')
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
    