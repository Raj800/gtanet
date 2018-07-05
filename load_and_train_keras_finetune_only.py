import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
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
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(85, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

txtfile = 'data_finetune_only.txt'

NAME = 'gtanet_binary_cropview_finetune_only'
MODEL_NAME = NAME+'.h5'
PREV_MODEL = NAME+'.h5'
LOAD_MODEL = True

if LOAD_MODEL:
    # load json and create model
    json_file = open(NAME+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(NAME+".h5")
    print("Loaded model from disk")

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:220]:
    layer.trainable = False
for layer in model.layers[220:]:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='categorical_crossentropy')


print("Starting Training...")
batch_count = 0
# if True:
try:
    epochs = 6
    for i in range(1, 2):
        count = 0
        print('----------- On Epoch: ' + str(i) + ' ----------')
        for x_train, y_train, x_test, y_test, batch_size, batch_count in load_batches(epoch=i, samples_per_batch=4000,txtfile=txtfile):
            # Model input requires numpy array
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            # Classification to one-hot vector
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            tensorboard = TensorBoard(log_dir="logs_finetune_only/{}".format(time.time()))

            # Fit model to batch


            # we need to recompile the model for these modifications to take effect
            # we use SGD with a low learning rate

            # we train our model again (this time fine-tuning the top 2 inception blocks
            # alongside the top Dense layers
            print('Fine-tuning the top 2 inception blocks alongside the top Dense layers')
            model.fit(x_train, y_train, batch_size=100, epochs=5,
                      validation_data=(x_test, y_test),
                      callbacks=[tensorboard]
                      )
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

        epochs += 1

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
