
# GTAnet
Simple tools that I used to generate and run a self-driving car model with VPilot and DeepGTAV.
[Link to Video](https://youtu.be/jQaX2iQSeMY)

## Requirements
1. [DeepGTAV](https://github.com/ai-tor/DeepGTAV) must be installed and [VPilot](https://github.com/cpgeier/VPilot) must be in the same folder.

2. Keras, Tensorflow, Numpy, h5py

## Files

- dataset.py - Uses in-game AI to drive car. Captures screen and saves driving variables in a pickle file
- drive_categorical.py - Excecutes a trained model on captured frames
- load_and_train_keras.py - Continues training a model on a dataset using Fine Tuning technique on Inception V3 model
- load_and_train_keras.py - Continues training a model on a dataset using Fine Tuning technique on Inception V3 model
- load_and_train_fromstart.py - Continues training a model on a dataset based on Inception V3 model from scratch
- models.py - Contains all the various models
- pickling.py - A simple program that displays frames from a pickled file
- preprocessing.py - Loads batches of frames and processes them from a pickled file to be used in training


## Training

The dataset I generated is around 216 gigabytes after running for 15 hours, so I would advise having plenty of space on a hard drive to store data.

## Improvements

If you would like to contribute I have some ideas on how this model could be improved:

- Dataset balancing - Most of the steering values will be around 0, with few around -1 and 1, so balancing the dataset would probably improve performance
- Better preprocessing

## Refrences

- https://psyber.io/
- https://github.com/sentdex/pygta5/
- https://github.com/aitorzip/DeepGTAV
- https://github.com/aitorzip/VPilot
- https://github.com/cpgeier/SantosNet
