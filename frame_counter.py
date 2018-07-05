import os
import numpy as np
import pickle
import gzip



'''
Use .pz file to generate a directory structure to train the inception v3.
This Code sorts all frames according to their steering angle. 
'''



data_dump_no = 1

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

straight_road = 0
curved_road = 0

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
        if 470 <= steering <= 530:
            curved_road += 1
        else:
            straight_road += 1
        count += 1
        if count % 100 == 0:
            print(count)

    except EOFError:
        data_dump_no += 1
        if data_dump_no > 16:
            break
        print('Opening dump.no {}'.format(data_dump_no))
        data = data_dump + str(data_dump_no) + '.pz'
        try:
            dataset = gzip.open(data)
        except Exception as e:
            print(e)
            if e == '[Errno 2] No such file or directory: \'{}\''.format(data_dump_no):
                print('Conversion Complete.'
                      ' Yay!!!!!!')
                break
    except Exception as e:
        print(e)
        data_dump_no += 1
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
print('Curved Road Frames =', curved_road)
print('Straight Road Frames =', straight_road)
print('Total Road Frames =', straight_road+curved_road)

