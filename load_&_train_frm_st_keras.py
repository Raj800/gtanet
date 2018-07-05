import tensorflow as tf
# config = tf   .ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
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

# create the base pre-trained model
base_model = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(35, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

txtfile = 'data_rgb.txt'
LOAD_NAME = 'gtanet_minimap_frm_st_keras1'
NAME = 'gtanet_minimap_frm_st_keras'
MODEL_NAME = NAME+'.h5'
PREV_MODEL = LOAD_NAME+'.h5'
LOAD_MODEL = False

if LOAD_MODEL:
    # load json and create model
    json_file = open(LOAD_NAME+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(LOAD_NAME+".h5")
    print("Loaded model from disk")
for j, layer in enumerate(base_model.layers):
    print(j, layer.name)


lr = 0.001

model.compile(optimizer=SGD(lr=lr, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
#
# for j, layer in enumerate(model.layers):
#     print(j, layer.name)

print("Starting Training...")
batch_count = 0
# if True:
try:
    for i in range(1, 6):  # 6 epochs done!
        count = 0
        print('----------- On Epoch: ' + str(i) + ' ----------')
        f = open(txtfile, 'w+')
        data_dump_no = 6
        f.write(str(data_dump_no))
        f.close()
        for x_train, y_train, x_test, y_test, batch_size, batch_count in load_batches(epoch=i, samples_per_batch=4000,
                                                                                      txtfile=txtfile):
            # Model input requires numpy array
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            # Classification to one-hot vector
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            tensorboard = TensorBoard(log_dir="logs/{}-{}-added_data-turn-only-v9.2".format(NAME, i))

            # Fit model to batch
            print('Training......')
            # train the model on the new data for a few epochs
            model.fit(x_train, y_train, batch_size=50, epochs=6,
                      validation_data=(x_test, y_test),
                      callbacks=[tensorboard])
            x_train = []
            y_train = []
            x_test = []
            y_test = []

            print('SAVING MODEL!')
            # serialize model to JSON
            model_json = model.to_json()
            with open(NAME + ".json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(NAME + ".h5")
            print("Saved model to disk")

            if batch_count % 2 == 0:
                print('SAVING MODEL!')
                # serialize model to JSON
                model_json = model.to_json()
                with open(NAME + "-1.json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(NAME + "-1.h5")
                print("Saved model to disk")

            if (batch_count + 1) % 2 == 0:
                print('SAVING MODEL!')
                # serialize model to JSON
                model_json = model.to_json()
                with open(NAME + "-2.json", "w") as json_file:
                    json_file.write(model_json)
                # serialize weights to HDF5
                model.save_weights(NAME + "-2.h5")
                print("Saved model to disk")

            count += 1

except Exception as e:
    print('Excepted with ' + str(e))
    print('SAVING MODEL!')
    # serialize model to JSON
    model_json = model.to_json()
    with open(NAME+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(NAME+".h5")
    print("Saved model to disk")
