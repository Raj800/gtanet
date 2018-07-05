import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from binarization_utils import binarize
from grabscreen import grab_screen
import time
from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy, Config, Dataset
from deepgtav.client import Client
from preprocessing import crop_bottom_half, minimap_processing, perspective_transform
from truncate import detruncate
from scipy.misc import imresize
from keras.utils import np_utils
import cv2
import numpy as np

NAME = './weights/acc=0.32/gtanet_minimap_frm_st_keras'
# NAME = 'gtanet_minimap_frm_st_keras'

smoothed_angle = 0
img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

print("Loading Model...")
# load json and create model
json_file = open(NAME+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(NAME+".h5")
print("Loaded model from disk")
print("Model Loaded. Compiling...")
model.compile(optimizer='Adadelta', loss='mean_squared_error')
'''
if input("Continue?") == "y": # Wait until you load GTA V to continue, else can't connect to DeepGTAV
    print("Conintuing...")
'''
# Loads into a consistent starting setting 
print("Loading Scenario...")
client = Client(ip='localhost', port=8000)  # Default interface
dataset = Dataset(rate=60, frame=[800, 600], throttle=True, brake=True, steering=True, location=True,
                          drivingMode=True)
scenario = Scenario(weathers='EXTRASUNNY', vehicle='blista', times=[12, 0],
                    drivingMode=-1) #, location=[-2573.13916015625, 3292.256103515625, 13.241103172302246])
client.sendMessage(Start(scenario=scenario))

count = 0
print("Starting Loop...")
while True:
    try:
        lasttime = time.time()
        # Collect and preprocess image
        message = client.recvMessage()
        image = message['frame']
        image = grab_screen(region=(0, 0, 800, 600))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = frame2numpy(image, (800, 600))
        image = minimap_processing(image)
        # print('minimap processed')
        # cv2.imshow('minimap', image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        image = (image / 255 - .5) * 2
        # Corrects for model input shape
        model_input = []
        model_input.append(image)

        # Converts classification to float for steering input
        nn_output = model.predict(np.array(model_input))

        try:
            steering_prediction = np.argmax(nn_output)
            steering_probability = np.amax(steering_prediction)
            steering_prediction = (detruncate(steering_prediction) - 500) / 500
        except Exception as e:
            steering_prediction = 0

        # throttle_prediction = np.argmax(nn_output[36:85])
        # throttle_prediction = throttle_prediction / 50
        #
        # brake_prediction = np.argmax(nn_output[36:85])
        # brake_prediction = brake_prediction / 50
        duration = time.time() - lasttime
        print('Predicted Steering Value :', str(steering_prediction))
        print('Prediction Probability :', str(steering_probability))
        if steering_prediction>0.08:
            print('Right')
        elif steering_prediction<-0.05:
            print('Left')
        else:
            print('Straight')
        # print('FPS :',  str(1/(duration)))
        # Mutiplication scales decimal prediction for harder turning

        client.sendMessage(Commands(0.3, 0.0, 2/(1+2.718**(-5.5*steering_prediction)) - 1))
        # client.sendMessage(Commands(0.3, 0.0, steering_prediction))
        count += 1

        degrees = (2/(1+2.718**(-5.5*steering_prediction)) - 1) * 180
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    except Exception as e:
        print("Excepted as: " + str(e))
        continue

cv2.destroyAllWindows()
client.sendMessage(Stop()) # Stops DeepGTAV
client.close()
