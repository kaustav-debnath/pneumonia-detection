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
# from tensorflow.keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasClassifier
# from tensorflow.keras.layers import Input,GlobalAveragePooling2D, ZeroPadding2D, Convolution2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten, Activation 
# from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.models import Model,load_model
# from tensorflow.keras.applications.imagenet_utils import decode_predictions 
# from tensorflow.keras.applications import MobileNet, VGG16
# from tensorflow.keras.applications.inception_v3 import InceptionV3

# from tensorflow.keras.applications.mobilenet import preprocess_input
# from sklearn.model_selection import GridSearchCV


# prefix = '/opt/ml/'
prefix = './'

IMG_ACTUAL_SIZE = 1024
input_path = os.path.join(prefix , 'input/data/')
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

training_path = os.path.join(input_path, 'stage_2_train_images/')
training_labels_path = input_path + 'stage_2_train_labels.csv'

print(training_path)
print(training_labels_path)

def create_model(X_train, y_train):
    classifier = load_model(os.path.join(model_path,'mobilenet_model.h5'))

    # Compiling the MobileNet
    classifier.compile(
        optimizer='Adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # declare the callbacks
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        # ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
        ModelCheckpoint(os.path.join(model_path,'model-pneumonia-classification.h5'), verbose=1, save_best_only=True, save_weights_only=True)
    ]
    # Fit gridsearch to training set
    history = classifier.fit(
        X_train,
        y_train,
        batch_size=50, 
        epochs=20, 
        callbacks=callbacks,
        validation_split = 0.2
    )
    return history

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

def test_images(data,img_width,img_height):
    for key in tqdm(data):
            res = load_image(data[key]['image_path'],img_width,img_height)
            print('patientId==>',key)
            print('target==>',data[key]["target"])
            draw_image_with_boxes(res,data[key]["bboxes"],(IMG_ACTUAL_SIZE/img_width))
            print('End Iterator-->',key,'--------------------------------------------------')

# create a dictionary of metadata info from training labels file
def create_dictionary(data):
    metainfo = {}
    for index, row in tqdm(data.iterrows()): 
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
    try:
        df_train_labels = pd.read_csv(training_labels_path)
        df_train_labels = df_train_labels.fillna(0) #data imputation replacing NAN values with 0s
        metainfo = create_dictionary(df_train_labels)
        test_images(metainfo,128,128)
        X, y = prepare_data(metainfo,128,128)
        classifier = create_model(X, y)
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
    return classifier
            
if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
    