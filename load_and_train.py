import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
from models import inception_v3 as googlenet
import numpy as np
import h5py
from preprocessing import load_batches


WIDTH = 299
HEIGHT = 299
LR = 1e-3
EPOCHS = 30

MODEL_NAME = 'gtanet.model'
PREV_MODEL = 'D:/Utilities/Raj/Project/Project/SantosNet-master/gtanet.model'
LOAD_MODEL = False


print("Loading Model ...")
model = googlenet(WIDTH, HEIGHT, 3, LR, output=35)
print("Model Loaded.")
'''
if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')
'''
txtfile = 'data_rgb.txt'

print("Starting Training...")
batch_count = 0
#try:

for i in range(1, 10):
    count = 0
    print('----------- On Epoch: ' + str(i) + ' ----------')
    f = open(txtfile, 'w+')
    data_dump_no = 1
    f.write(str(data_dump_no))
    f.close()
    for x_train, y_train, x_test, y_test, samples_per_batch, batch_count in load_batches(epoch=i, txtfile=txtfile, samples_per_batch=1000):
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