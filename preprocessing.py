import os
import numpy as np
import pickle
import cv2
import gzip
from scipy.misc import imresize
from deepgtav.messages import frame2numpy
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from truncate import truncate
from binarization_utils import binarize

minimap_x1 = 37
minimap_y1 = 490
minimap_x2 = 193
minimap_y2 = 590


# define range of blue color in HSV
lower_purple = np.array([265/2, 76.5, 90])
upper_purple = np.array([285/2, 255, 250])

def perspective_transform(img):
    h, w = img.shape[:2]
    src = np.float32([[w, h*0.8],
                      [0, h*0.8],
                      [300, h*0.59],
                      [500, h*0.59]])
    dst = np.float32([[700, h],       # br
                      [100, h],       # bl
                      [100, 0],       # tl
                      [700, 0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)
    return warped


def crop_bottom_half(image):
    # Crops to bottom half of image
    return image[int(image.shape[0] * 0.5):image.shape[0]]

def minimap_processing(image):
    minimap = image[minimap_y1:minimap_y2, minimap_x1:minimap_x2]
    minimap_hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(minimap_hsv, lower_purple, upper_purple)

    # Bitwise-AND mask and original image
    minimap_purple = cv2.bitwise_and(minimap, minimap, mask=mask)
    minimap_purple[563 - minimap_y1:575 - minimap_y1, 110 - minimap_x1:120 - minimap_x1] = image[563:575, 110:120]
    minimap_purple = imresize(minimap_purple, (265, 800), interp='nearest')
    image[0:minimap_purple.shape[0], 0:minimap_purple.shape[1]] = minimap_purple
    image = imresize(image, (299, 299))

    return image


def load_batches(epoch, txtfile, verbose=1, samples_per_batch=4000):
    ''' Generator for loading batches of frames'''

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

    # y = []  # outputs
    x = []  # input
    s = []  # steering
    # t = []  # throttle
    # b = []  # brake
    count = 0

    while True:
        try:
            print('----------- On Epoch: ' + str(epoch) + ' ----------')
            print('----------- On Batch: ' + str(batch_count) + ' ----------')
            while count < samples_per_batch:
                data_dict = pickle.load(dataset)
                steering = int(float(data_dict['steering']) * 750) + 500
                if steering >= 1000:
                    steering = 999
                if steering < 0:
                    steering = 0
                if 470 <= steering <= 530:
                    continue

                image = frame2numpy(data_dict['frame'], (800, 600))
                image = minimap_processing(image)
                # cv2.imshow('original', image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                image = (image / 255 - .5) * 2

                x.append(image)

                # Steering in dict is between -1 and 1, scale to between 0 and 999
                # and then 0 to 34 for categorical input
                if steering >= 1000:
                    steering = 999
                if steering < 0:
                    steering = 0
                steering = truncate(steering)
                s.append(steering)

                # # Throttle in dict is between 0 and 1, scale to between 0 and 49 for categorical input
                # throttle = int(data_dict['throttle'] * 50)
                # if throttle >= 50:
                #     throttle = 49
                # if throttle < 0:
                #     throttle = 0
                # t.append(throttle)
                #
                # # brake in dict is between 0 and 1, scale to between 0 and 49 for categorical input
                # brake = int(data_dict['brake'] * 50)
                # if brake >= 50:
                #     brake = 49
                # if brake < 0:
                #     brake = 0
                # b.append(brake)

                count += 1
                if (count % 250) == 0 and verbose == 1:
                    print('  ' + str(count) + ' data points loaded in batch.')
            count = 0
            print('Batch loaded.')
            s = np_utils.to_categorical(s, num_classes=35)
            # t = np_utils.to_categorical(t, num_classes=50)
            # b = np_utils.to_categorical(b, num_classes=50)

            # y = np.hstack([s, b])

            # Train test split
            x_train, x_test, y_train, y_test = train_test_split(x, s, test_size=0.2, random_state=1, shuffle=True)

            # y = []
            x = []
            s = []  # steering
            # t = []  # throttle
            # b = []  # brake

            batch_count += 1
            yield x_train, y_train, x_test, y_test, samples_per_batch, batch_count
        except EOFError:

            data_dump_no += 1
            f = open(txtfile, 'w+')
            f.write(str(data_dump_no))
            f.close()
            print('Opening dump.no {}'.format(data_dump_no))
            data = data_dump + str(data_dump_no) + '.pz'
            if os.path.exists(data):
                dataset = gzip.open(data)
                pass
            else:
                print('File Completed')
                break
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
            if os.path.exists(data):
                dataset = gzip.open(data)
                pass
            else:
                print(e)
                print('Conversion Complete.'
                      ' Yay!!!!!!')

                break
            pass
