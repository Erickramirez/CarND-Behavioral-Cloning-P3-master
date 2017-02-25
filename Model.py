import csv
import numpy as np
from collections import defaultdict
import matplotlib.image as mpimg
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from random import randint
from PIL import Image
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import Input, ELU

'''Read csv file'''
def read_csvfile (csv_filename):
    csv_data=defaultdict(list)
    freader = csv.reader(open(csv_filename), delimiter=',')
    header = next(freader)
    for row in freader:
        for index in range(len(header)):
            csv_data[header[index]].append(''.join(row[index]).strip() )
    return csv_data

def AOI_image (img, resize_value):
    #Area of interest of the image, it applies crop and resize.
    (h, w) = img.shape[:2]
    y1=int(h*0.375)
    y2=int(h*0.125)
    x=0
    nimg = cv2.resize(img[y1:h-y2, x:x+w],resize_value)
    return nimg

def read_image (data_line, steering_shift, resize_image):
    image_elected_id = np.random.randint(3)
    img = mpimg.imread(data_line[column_name[image_elected_id]] )
    img = AOI_image(img, resize_image)
    if column_name[image_elected_id] == 'center':
        steering = data_line['steering']
    elif column_name[image_elected_id] == 'left':
        steering = data_line['steering'] + steering_shift
        if steering>1:
            steering = 1.0 
    else:
        steering = data_line['steering']  - steering_shift
        if steering<-1:
            steering = -1.0        
    return img , steering
	
def image_flip(img,steering):
    nimg = cv2.flip(img,1)
    nsteering =  -1* steering
    return nimg,nsteering

def image_traslation (img,steering, x,y):
    (h, w) = img.shape[:2]
    M = np.float32([[1,0,x],[0,1,y]])
    nimg = cv2.warpAffine(img,M,(w,h))
    nsteering = steering+ (x/125)#200
    return nimg,nsteering

def change_brightness(img, brightness_value):  
    # convert to HSV to change the brightness 
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * brightness_value
    #Convert to RGB
    nimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return nimg

def gray_scale (img):
    nimg =cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #nimages = np.asarray(nimages, dtype=np.float32)
    return nimg

def perturb_steering(steering):
    nsteering= steering*(1+np.random.uniform(-1, 1)/50)
    return nsteering
       
     
def preprocess_image(data_line, resize_image):
    img, steering = read_image(data_line, 0.25,resize_image)   
    if np.random.randint(2) == 1:
        img, steering = image_flip(img,steering)
    if np.random.randint(2) == 1:
        img,steering=image_traslation (img,steering, randint(-25,25) ,0)
    if np.random.randint(2) == 1:
        img=change_brightness(img, np.random.uniform(0.6, 1.3))
    img =np.asarray(img, dtype=np.float32)
    steering = perturb_steering(steering)
    return img, steering
	
def generate_train_batch(img_dictionary, w,h, batch_size = 64):
    batch_images = np.zeros((batch_size, h, w,3))
    batch_steering = np.zeros((batch_size))
    d_line = {}
    while 1:
        index_batch = np.random.choice(len(img_dictionary['steering']), batch_size)
        for index in range(batch_size) :
            d_line['center'] = img_dictionary['center'][index_batch[index]]
            d_line['left'] = img_dictionary['left'][index_batch[index]]
            d_line['right'] = img_dictionary['right'][index_batch[index]]
            d_line['steering'] =  img_dictionary['steering'][index_batch[index]]
            Ximg,ysteering = preprocess_image(d_line, (w,h))
            batch_images[index] =Ximg
            batch_steering[index] =ysteering
        yield batch_images, batch_steering

def save_model(json_file,weights_file):
    if Path(json_file).is_file():
        os.remove(json_file)
    json_model = model.to_json()
    
    with open(json_file,"w" ) as w_file:
        w_file.write(json_model)
        
    if Path(weights_file).is_file():
        os.remove(weights_file)
    print (json_file + ' and '+weights_file + ' have been saved successfully' )
    model.save_weights(weights_file)


		
column_name =['center','left', 'right']

driving_log = read_csvfile('driving_log.csv')
driving_log['steering'] = np.array(driving_log['steering'],dtype=np.float32)
driving_log['throttle'] = np.array(driving_log['throttle'],dtype=np.float32)
driving_log['brake'] = np.array(driving_log['brake'],dtype=np.float32)
driving_log['speed'] = np.array(driving_log['speed'],dtype=np.float32) 

valid_indexes =driving_log['speed']>10 #removing data with no speed

for _key in (list(driving_log.keys())):
    driving_log[_key] =np.array (driving_log[_key])
    driving_log[_key] =driving_log[_key] [valid_indexes]

    
valid_indexes = driving_log['throttle'] >0.2#removing data with no speed
for _key in (list(driving_log.keys())):
    driving_log[_key] =np.array (driving_log[_key])
    driving_log[_key] =driving_log[_key] [valid_indexes]
    

valid_indexes = driving_log['throttle'] >0.2#removing data with no speed
for _key in (list(driving_log.keys())):
    driving_log[_key] =np.array (driving_log[_key])
    driving_log[_key] =driving_log[_key] [valid_indexes]
    
for index in range (len(driving_log['steering'])):
    if driving_log['steering'][index]>1:
        driving_log['steering'][index] = 1
    elif driving_log['steering'][index]<-1:
        driving_log['steering'][index] = -1

train_data_generator = generate_train_batch(driving_log, 200,66,256)
validation_data_generator =  generate_train_batch(driving_log, 200,66,256)

def get_NVIDIA_model(): #https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    input_shape = (66, 200,3)
    filter_size = 3
    pool_size = (2,2)
    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))
    model.add(Convolution2D(24,5, 5, subsample=(2, 2),init='he_normal',border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(36,5, 5, subsample=(2, 2),init='he_normal',border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(48,5, 5,  subsample=(2, 2),init='he_normal',border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64,3, 3, subsample=(1, 1), init='he_normal',border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(64,3, 3, subsample=(1, 1), init='he_normal',border_mode='valid'))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dense(10,init='he_normal'))
    model.add(ELU())
    model.add(Dropout (0.5))
    model.add(Dense(1, init='he_normal'))
       
    return model 

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model =get_NVIDIA_model()
model.compile(optimizer=adam,loss='mse')


sample_size = len(driving_log['center'] )
epoch_size = sample_size*8
for index in range(5):
    history = model.fit_generator(train_data_generator, validation_data = validation_data_generator,samples_per_epoch = epoch_size, nb_val_samples = sample_size, nb_epoch=2) 
    json_file = 'model_' + str(index) + '.json'
    weights_file = 'model_' + str(index) + '.h5'
    save_model(json_file,weights_file )
	