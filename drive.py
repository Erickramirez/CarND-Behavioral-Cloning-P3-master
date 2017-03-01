import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops


sio = socketio.Server()
app = Flask(__name__)
model = None

def AOI_image (img, resize_value):
    #Area of interest of the image, it applies crop and resize.
    (h, w) = img.shape[:2]
    y1=int(h*0.35)
    y2=int(h*0.16)
    x=0
    nimg = cv2.resize(img[y1:h-y2, x:x+w],resize_value)
    #nimg = cv2.cvtColor(nimg,cv2.COLOR_RGB2YUV)
    return nimg

def preprocess_input(img):
    return AOI_image (img,(200,66))
	

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    # model >= 5
    x = preprocess_input(np.array(image))
    image_array = np.asarray(x, dtype=np.float32)
    transformed_image_array = image_array[None, :, :, :]

    

    speed = float(speed)
    steering_angle = float(model.predict(transformed_image_array, batch_size=1)) 
    throttle_max = 1.0
    throttle_min = -1.0
    #steering_threshold = 3./25

    # Targets for speed controller
    nominal_set_speed = 30
    steering_set_speed = 30

    K = 0.35   # Proportional gain

    # Slow down for turns
    if abs(steering_angle) > 0.2:
        throttle = 0.15
    elif  abs(steering_angle) < 0.05:
        throttle = 0.4
    else:
        throttle = 0.25
    #    set_speed = steering_set_speed
    #else:
    #    set_speed = nominal_set_speed
    if speed>20:
        throttle = 0
    elif  speed < 8:
        throttle = 0.8
    
    if  speed > 14 and abs(steering_angle) >= 0.20:
        throttle = -1
    #throttle = (set_speed - speed)*K
    #throttle = min(throttle_max, throttle)
    #throttle = 0.3
#throttle = 0.2
    # else don't change from previous
    print("{0:.2f}".format(steering_angle), "--",throttle,"--", speed )
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # model = model_from_json(json.load(jfile))
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
