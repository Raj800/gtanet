import os
import numpy as np
import random
import pickle
import gzip
from deepgtav.messages import frame2numpy
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import cv2
import seaborn as sns
import time
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from preprocessing import perspective_transform
from moviepy.editor import *
from scipy.misc import imresize



'''
Use .pz file to generate a directory structure to train the inception v3.
This Code sorts all frames according to their steering angle. 
'''


file_name = 'dataset'
if not os.path.isdir(file_name):
    os.mkdir(file_name)

txtfile = 'dataset/data.txt'
if os.path.exists(txtfile):
    f = open(txtfile, 'r')
    data_dump_no = int(f.read())
    f.close()
else:
    f = open(txtfile, 'w+')
    data_dump_no = 1
    f.write(str(data_dump_no))
    f.close()

data_dump = 'dataset_minimap_800x600-'
data = data_dump + str(data_dump_no) + '.pz'
print(data)
print('Opening dump.no {}'.format(data_dump_no))
dataset = gzip.open(data)
batch_count = 0

training_data = []
count = 0
starting_value = 0
np_value = 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (800, 600))



'''

########################## Creates folders named 0 to 999 ################################
for i in range(0,1000):
    file_name = 'dataset/{}'.format(i)
    os.mkdir(file_name)
print('Directories generated.'
      'Now Storing images to respective directories')
'''


while True:
    try:

        data_dct = pickle.load(dataset)
        steering = int(float(data_dct['steering']) * 500) + 500

        frame = data_dct['frame']
        image = frame2numpy(frame, (800, 600))
        # image = ((image / 255) - .5) * 2  # Simple preprocessing





        ################# For saving data in a single folder and steering in text file #############
        file_name = 'driving_dataset/'
        if not os.path.isdir(file_name):
            os.mkdir(file_name)
        file_name = 'driving_dataset/{}.jpg'.format(starting_value)
        str = '{}.jpg {}\r'.format(starting_value, round(float(data_dct['steering'])*180), 5)
        cv2.imwrite(file_name, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        with open("driving_dataset/data.txt", "a+") as text_file:
            text_file.write(str)



        # make video from images
        # out.write(image)



        if starting_value % 100 == 0:
            print('Read', starting_value, 'frames.')



        '''
        ################### for seperating data into labelled folders ##############################

        # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
        steering = int(float(data_dct['steering']) * 500) + 500
        if steering == 1000:
            steering = 999

        file_name = 'dataset/{}'.format(steering)
        if not os.path.isdir(file_name):
            os.mkdir(file_name)

        file_name = 'dataset/{}/img-{}.jpg'.format(steering, starting_value)
        cv2.imwrite(file_name, image, [cv2.IMWRITE_JPEG_QUALITY, 90])

        '''



        # training_data.append([image, steering])
        ############### To convert data to a numpy array with image and steering angle #########
        if len(training_data) == 200:
            np_filename = 'dataset/training_data-{}'.format(np_value)
            fname = np_filename + '.npy'
            while os.path.exists(fname):
                print('{} already present. Moving along,'.format(fname))
                np_value += 1
                np_filename = 'dataset/training_data-{}'.format(np_value)
                fname = np_filename + '.npy'

            np.save(np_filename, training_data)
            print('SAVED', np_filename)

            training_data = []
            np_value += 1


        starting_value += 1


    except EOFError:
        data_dump_no += 1
        f = open(txtfile, 'w+')
        f.write(str(data_dump_no))
        f.close()
        print('Opening dump.no {}'.format(data_dump_no))
        data = data_dump + str(data_dump_no) + '.pz'
        try:
            dataset = gzip.open(data)
        except Exception as e:
            print(e)
            if e == '[Errno 2] No such file or directory: \'{}\''.format(data_dump_no):
                print('Conversion Complete.'
                      ' Yay!!!!!!')
    except Exception as e:
        print(e)
        data_dump_no += 1
        f = open(txtfile, 'w+')
        f.write(str(data_dump_no))
        f.close()
        print('Data of this file is corrupted after this point. '
              'Don\'t worry, Opening next file')
        print('Opening dump.no {}'.format(data_dump_no))
        data = data_dump + str(data_dump_no) + '.pz'
        try:
            dataset = gzip.open(data)
        except Exception as e:
            print(e)
            print('Conversion Complete.'
                  ' Yay!!!!!!')
            break
        pass





