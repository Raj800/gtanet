import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
import numpy as np
import keras
from keras.callbacks import TensorBoard
import time
from keras.models import model_from_json

from preprocessing import load_batches

from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from models import inception_v3 as googlenet
from random import shuffle


FILE_I_END = 34
WIDTH = 299
HEIGHT = 299
LR = 1e-3
EPOCHS = 30

txtfile = 'data_frm_start_rgb.txt'

NAME = 'gtanet_topview_frmstart'
MODEL_NAME = NAME+'.model'
PREV_MODEL = NAME+'.model'

LOAD_MODEL = True

model = googlenet(WIDTH, HEIGHT, 3, LR, output=35, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')

count = 0

for epoch in range(EPOCHS):
    try:
        for X, Y, test_x, test_y, batch_size, batch_count in load_batches(epoch=epoch,txtfile=txtfile):
            model.fit({'input': X}, {'targets': Y}, n_epoch=6,
                  validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=2500,
                  show_metric=True, run_id=MODEL_NAME)
            print('SAVING MODEL!')
            model.save(MODEL_NAME)
            print("Saved model to disk")
            if batch_count % 2 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                print("Saved model to disk")

            if (batch_count + 1) % 2 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)
                print("Saved model to disk")

        count += 1


    except Exception as e:
        print(str(e))
        print('SAVING MODEL!')
        model.save(MODEL_NAME)
        if batch_count % 2 == 0:
            print('SAVING MODEL!')
            model.save(MODEL_NAME)
            print("Saved model to disk")

        if (batch_count + 1) % 2 == 0:
            print('SAVING MODEL!')
            model.save(MODEL_NAME)
            print("Saved model to disk")
    f = open(txtfile, 'w+')
    data_dump_no = 1
    f.write(str(data_dump_no))
    f.close()