import os
import numpy as np
import random
import pickle
import cv2
import gzip
from scipy.misc import imresize
from deepgtav.messages import frame2numpy
from keras.utils import np_utils
from sklearn.model_selection import train_test_split



def perspective_transform(img):

    h, w = img.shape[:2]
    src = np.float32([[w,480],
                      [0, 480],
                      [320, 315],
                      [480, 315]])
    dst = np.float32([[w, h],       # br
                      [0, h],       # bl
                      [0, 0],       # tl
                      [w, 0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped

def crop_bottom_half(image):
    # Crops to bottom half of image
    return image[int(image.shape[0] / 2.25):image.shape[0]]


def load_batches(epoch, verbose=1, samples_per_batch=1500):
    ''' Generator for loading batches of frames'''
    dataset = gzip.open('dataset_test_320x160.pz')
    batch_count = 0

    while True:
        try:
            x = []
            y = []
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            y_train = []
            y_val = []
            count = 0
            print('----------- On Epoch: ' + str(epoch) + ' ----------')
            print('----------- On Batch: ' + str(batch_count) + ' -----------')
            while count < samples_per_batch:
                data_dct = pickle.load(dataset)
                frame = data_dct['frame']
                image = frame2numpy(frame, (320, 160))
                # image = crop_bottom_half(image)
                image = ((image / 255) - .5) * 2  # Simple preprocessing
                # Train test split
                # TODO: Dynamic train test split | Test series at end of batch
                x.append(image)
                # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
                temp = int(float(data_dct['steering']) * 500) + 500
                if temp >= 1000:
                    temp = 999
                if temp < 0:
                    temp = 0

                y.append(temp)
                count += 1
                # if (count % 250) == 0 and verbose == 1:
                # print('     ' + str(count) + ' data points loaded in batch.')
            print('Batch loaded.')
            y = np_utils.to_categorical(y, num_classes=1000)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)

            batch_count += 1
            yield x_train, y_train, x_test, y_test, samples_per_batch
        except EOFError:  # Breaks at end of file
            break
        except Exception as e:
            print(e)
            print('Error reading the next block! The file after this point is corrupted Its OK. Skipping it.........')
            break


def load_from_npy(epoch, batch_number, batch_count):
    file_name = 'dataset/training_data-{}.npy'.format(batch_count)
    batch_count = 1
    iteration = 1
    image_list = []
    print('----------- On Epoch: ' + str(epoch) + ' -----------')
    print('----------- On Batch: ' + str(batch_count) + ' -----------')

    print('Opening training_data-{}.npy'.format(batch_number))
    file_name = 'dataset/training_data-{}.npy'.format(batch_number)
    batch_count += 1
    if os.path.isfile(file_name):
        print('File exists, moving along')
        data_dct = np.load(file_name)
    else:
        print('File does not exist Exitting!')
        return None

    frame_list = np.array([i[0] for i in data_dct]).reshape(-1, 600, 800, 3)
    steering_list = [i[1] for i in data_dct]
    data_dct = []

    # Steering in dict is between -1 and 1, scale to between 0 and 999 for categorical input
    for i in range(len(frame_list)):
        if steering_list[i] < 0:
            steering_list[i] = 0
        if steering_list[i] >= 1000:
            steering_list[i] = 999
        image = perspective_transform(frame_list[i])
        image = cv2.resize(image, (320, 240))
        image = ((image / 255) - .5) * 2  # Simple preprocessing
        image_list.append(image)
    frame_list = []
    steering_list = np_utils.to_categorical(steering_list, num_classes=1000)
    x_train, x_test, y_train, y_test = train_test_split(image_list, steering_list, test_size=0.2,
                                                        random_state=1, shuffle=True)
    steering_list = image_list = []
    return x_train, y_train, x_test, y_test


