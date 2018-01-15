import os
from moviepy.editor import *
import os
import numpy as np
import random
import pickle
import gzip
from deepgtav.messages import frame2numpy
import cv2
import seaborn as sns
import time
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from preprocessing import perspective_transform
from scipy.misc import imresize



'''
Use .pz file to generate a directory structure to train the inception v3.
This Code sorts all frames according to their steering angle. 
'''


data_dump_no = 1
data_dump = 'dataset_test_800x600-'
data = data_dump + str(data_dump_no) + '.pz'
dataset = gzip.open(data)

batch_count = 0

training_data = []
count = 0
starting_value = 1
np_value = 1

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (800, 600))

img = cv2.imread('steering_wheel_image.jpg')
img = imresize(img, (150,150))
rows = cols = 150
smoothed_angle = 1
file_name = 'dataset/training_data-{}.npy'.format(data_dump_no)

'''

########################## Creates folders named 0 to 999 ################################
for i in range(0,1000):
    file_name = 'dataset/{}'.format(i)
    os.mkdir(file_name)
print('Directories generated.'
      'Now Storing images to respective directories')
'''




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
#
# video_output = "project_video_output-1.mp4"
# clip = VideoFileClip("challenge.mp4")
# new_clip = clip.fl_image(perspective_transform)
# new_clip.write_videofile(video_output, audio=False)

def loadfile(data_dump_no_1):
    if starting_value >= 1500 or starting_value == 1:
        print('Opening training_data-{}.npy'.format(data_dump_no))
        file_name = 'dataset/training_data-{}.npy'.format(data_dump_no)
        data_dump_no_1 += 1
        if os.path.isfile(file_name):
            print('File exists, moving along')
            return np.load(file_name), data_dump_no_1
        else:
            print('File does not exist Exitting!')

while True:
    data_dct, data_dump_no = loadfile(data_dump_no)
    frame_list = []
    steering_list = []
    frame_list = np.array([i[0] for i in data_dct]).reshape(-1, 600, 800, 3)
    steering_list = [i[1] for i in data_dct]
    data_dct , data_dump_no
    # image = ((image / 255) - .5) * 2  # Simple preprocessing
    for i in range(len(frame_list)):
        starting_value += 1
        image = frame_list[i]
        steering = (steering_list[i]-500)/500 *180
        smoothed_angle += 0.2 * pow(abs((steering - smoothed_angle)), 2.0 / 3.0) * (steering - smoothed_angle) / abs(
            steering - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        topview = perspective_transform(image)
        # out.write(topview)
        cv2.putText(img=topview, text=str(steering), org=(600, 100),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(0, 255, 0))
        x_offset = 600
        y_offset = 300
        topview[y_offset:y_offset + img.shape[0], x_offset:x_offset + img.shape[1]] = dst
        topview = imresize(topview, (270, 480))
        cv2.imshow('bird eye view', topview)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    starting_value = 1
    if data_dump_no == 4:
        break
    # except Exception as e:
    #     print(e)
    #     pass
out.release()
