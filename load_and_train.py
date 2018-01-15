import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
from models import inception_v3 as googlenet
import numpy as np
import h5py
from preprocessing import load_batches


WIDTH = 160
HEIGHT = 320
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'gtanet.model'
PREV_MODEL = 'D:/Utilities/Raj/Project/Project/SantosNet-master/gtanet.model'
LOAD_MODEL = True


print("Loading Model ...")
model = googlenet(WIDTH, HEIGHT, 3, LR, output=1000)
print("Model Loaded.")
'''
if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')
'''

print("Starting Training...")
batch_count = 0
#try:

for i in range(1, 10):
    count = 0
    print('----------- On Epoch: ' + str(i) + ' ----------')
    for x_train, y_train, x_test, y_test in load_batches():
        # Model input requires numpy array
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        # Classification to one-hot vector
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        # Fit model to batch
        '''
        def fit(self, X_inputs, Y_targets, n_epoch=10, validation_set=None,
                show_metric=False, batch_size=None, shuffle=None,
                snapshot_epoch=True, snapshot_step=None, excl_trainops=None,
                validation_batch_size=None, run_id=None, callbacks=[]):
        '''
        model.fit({'input': x_train}, {'targets': y_train}, n_epoch=1,
                  validation_set=({'input': x_test}, {'targets': y_test}),
                  snapshot_step=2500, show_metric=True, run_id=MODEL_NAME
                  )

        if count % 10 == 0:
            print('SAVING MODEL!')
            model.save(MODEL_NAME)

#except Exception as e:
#    print('Excepted with ' + str(e))
#    print('Saving model...')
#    model.save('model_trained_categorical.h5')
#    print('Model saved.')