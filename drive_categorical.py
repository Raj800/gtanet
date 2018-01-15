import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client
from preprocessing import crop_bottom_half

import cv2
import numpy as np


print("Loading Model...")
# load json and create model
json_file = open('gtanet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("gtanet.h5")
print("Loaded model from disk")
print("Model Loaded. Compiling...")
model.compile(optimizer='Adadelta', loss='mean_squared_error')
'''
if input("Continue?") == "y": # Wait until you load GTA V to continue, else can't connect to DeepGTAV
    print("Conintuing...")
'''
# Loads into a consistent starting setting 
print("Loading Scenario...")
client = Client(ip='localhost', port=8000) # Default interface
scenario = Scenario(weathers='EXTRASUNNY',vehicle='blista',times=[12,0],drivingMode=-1,location=[-2573.13916015625, 3292.256103515625, 13.241103172302246])
client.sendMessage(Start(scenario=scenario))

count = 0
print("Starting Loop...")
while True:
    try:    
        # Collect and preprocess image
        message = client.recvMessage()
        image = frame2numpy(message['frame'], (480,270))
        # cv2.imshow('img',image)
        image = ((image/255) - .5) * 2
        image = crop_bottom_half(image)

        # Corrects for model input shape
        model_input = []
        model_input.append(image)

        # Converts classification to float for steering input
        print('.')
        category_prediction = np.argmax(model.predict(np.array(model_input)))
        decimal_prediction = (category_prediction - 500) / 500
        print('Category: ' + str(category_prediction) + '     Decimal: ' + str(decimal_prediction))

        client.sendMessage(Commands(0.6, 0.0, decimal_prediction * 3)) # Mutiplication scales decimal prediction for harder turning
        count += 1
    except Exception as e:
        print("Excepted as: " + str(e))
        continue

client.sendMessage(Stop()) # Stops DeepGTAV
client.close()
