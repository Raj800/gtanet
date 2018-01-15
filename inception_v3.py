import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from keras.preprocessing import image
from preprocessing import load_batches
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import backend as K
from keras.optimizers import SGD
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
import numpy as np
from keras.utils import np_utils
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 1000 classes
predictions = Dense(1000, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

nb_epoch = 10

# prepare data augmentation configuration
for i in range(0, nb_epoch):
    print('----------- On Epoch: ' + str(i) + ' ----------')
    for x_train, y_train, x_test, y_test, x_val, y_val, batch_count in load_batches():
        # Model input requires numpy array
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        x_val = np.array(x_val)
        # Classification to one-hot vector
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        y_val = np.array(y_val)
        batch_count = batch_count

        #model.summary()


        # train the model on the new data for a few epochs
        model.fit(x=x_train, y=y_train,
                  epochs=1,
                  shuffle=True,
                  verbose=1,
                  validation_data=(x_val, y_val),
                  steps_per_epoch=10,
                  validation_steps=2
                  )

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        '''
        for i, layer in enumerate(base_model.layers):
           print(i, layer.name)
        '''
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in model.layers[:249]:
           layer.trainable = False
        for layer in model.layers[249:]:
           layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate

        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit(x=x_train, y=y_train,
                  epochs=1,
                  shuffle=True,
                  verbose=1,
                  validation_data=(x_val, y_val),
                  steps_per_epoch=10,
                  validation_steps=2
                  )
        if (batch_count % 20) == 0:
            print('Saving checkpoint ' + str(batch_count))
            model.save('model_checkpoint' + batch_count + '.h5')
            print('Checkpoint saved. Continuing...')
