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
from preprocessing import load_from_npy


FILE_I_END = 34
WIDTH = 240
HEIGHT = 320
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'gtanet.model'
PREV_MODEL = 'gtanet.model'

tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

LOAD_MODEL = False

model = googlenet(WIDTH, HEIGHT, 3, LR, output=1000, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')

for epoch in range(EPOCHS):
    data_order = [i for i in range(1, FILE_I_END + 1)]
    shuffle(data_order)
    for count, i in enumerate(data_order):
        try:
            X, Y, test_x, test_y = load_from_npy(epoch, i, count)
            model.fit({'input': X}, {'targets': Y}, n_epoch=1,
                      validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=2500,
                      show_metric=True, run_id=MODEL_NAME)

            if count % 5 == 0:
                print('SAVING MODEL!')
                model.save(MODEL_NAME)

        except Exception as e:
            print(str(e))
