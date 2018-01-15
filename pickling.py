# Simple pickle reading program; Displays image from dataset
import pickle
import gzip
import cv2
from deepgtav.messages import frame2numpy
import time


def crop_bottom_half(image):
    # Crops to bottom half of image
    return image[int(image.shape[0] / 1.8):image.shape[0]]


file = gzip.open('dataset_test.pz')
count = 0

while True:
    try:
        data_dict = pickle.load(file)  # Iterates through pickle generator
        count += 1
        # Every 10000 frames prints steering and displays frame
        if (count % 100) == 0:
            print(str(data_dict['steering']) + '	On Count: ' + str(count))
            frame = data_dict['frame']
            # Show full image
            image = frame2numpy(frame, (480, 270))
            cv2.imshow('img', image)
            #time.sleep(0.1)
            cv2.waitKey(-1)  # Must press q on keyboard to continue to next frame
            # Show cropped image
            image = crop_bottom_half(image)
            #cv2.imshow('img', image)
            #cv2.waitKey(-1)  # Must press q on keyboard to continue to next frame


    except EOFError:
        print('End of File')
        break
print(count)
