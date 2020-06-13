import os
import numpy as np
import cv2
import pydicom
import matplotlib
import matplotlib.pyplot as plt
import PIL
import flask
import boto3
from pydicom import dcmread
from pydicom.filebase import DicomBytesIO

from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from keras import backend as K
from keras.models import load_model
#from sklearn.preprocessing import StandardScaler
import boto3

app = flask.Flask(__name__)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model():
        """
        Get the model object for this instance,
        loading it if it's not already loaded.
        """
        print('inside get model')
        path = os.path.join(model_path, 'mobilenet-classification.h5')
        print('model path=>',path)
        print('prefix=>',os.listdir(prefix))
        print('model_path=>',os.listdir(model_path))
        model = load_model(path)
#         BUCKET_NAME = "s3://sagemaker-capstone-gl"
#         OBJECT_NAME = "model/sagemaker-2020-06-13-15-43-53-718/output/model.tar.gz"
#         print(BUCKET_NAME, OBJECT_NAME)
#         s3 = boto3.client('s3')
#         model_file = s3.download_file('BUCKET_NAME', 'OBJECT_NAME', 'FILE_NAME')
#         print(model_file)
#         os.system('tar -zxvf model.tar.gz')
        
#         print(os.listdir())
#         model = load_model('mobilenet-classification.h5')
#         config = model.get_config()
#         weights = model.get_weights()
#         print('config=>',config,'weights=>',weights)
        return model
        
def load_image(ds,img_width,img_height):
    #this method helps in reading .dcm file
    #ds = pydicom.dcmread(image_path)
    img = ds.pixel_array
    #calling cv2.resize to reduce the dimension of image
    res = cv2.resize(img, (img_width,img_height), interpolation=cv2.INTER_AREA)
    #converting the image array data into float 
    res = res.astype(np.float32)/255
    cv2.destroyAllWindows()
    return res
    
@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy.
    In this sample container, we declare it healthy if we can load the model
    successfully.
    """

    # Health check -- You can insert a health check here
    print('Inside ping')
    health = get_model() is not None
    status = 200 if health else 404
    return flask.Response(
        response='\n',
        status=status,
        mimetype='application/json')
        
@app.route('/invocations', methods=['POST'])
def upload_file():
    print('inside invocations')
    try:
        data = request.get_data()
        print('data=>',data)
        raw = DicomBytesIO(data)
        ds = dcmread(raw)
        print(ds.pixel_array)
        ds = load_image(ds, 128 , 128)
        print('ds.shape=>',ds.shape)
        print(ds)
        model = get_model()
        prediction = model.predict(ds)
        status = 200
    except Exception as e:
        print(e)
        return flask.Response(
            response= e,
            status=404,
            mimetype='text/plain')     
    return flask.Response(
        response = prediction,
        status=status,
        mimetype = 'application/json')

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port = 8080)
    