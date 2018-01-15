import os
import gzip
import pickle
import argparse

import cv2
import numpy as np

from deepgtav.messages import frame2numpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse .pz files')
    parser.add_argument('-d', '--dataset_path', default='', help='Place to store the dataset')
    parser.add_argument('--file_prefix', default='dataset_test', help='File to be parsed')
    parser.add_argument('--show', action='store_true', help='Display while parsing')
    args = parser.parse_args()

    save_folder = os.path.join(args.dataset_path, args.file_prefix)
    file_name = args.file_prefix + '.pz'
    config_name = args.file_prefix + '.pz'

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    '''
    if os.path.exists(os.path.join(args.dataset_path, file_name)):
        os.rename(os.path.join(args.dataset_path, file_name), os.path.join(save_folder, file_name))
    if os.path.exists(os.path.join(args.dataset_path, config_name)):
        os.rename(os.path.join(args.dataset_path, config_name), os.path.join(save_folder, config_name))
    '''

    with open(os.path.join(save_folder, config_name)) as f:
        hours = [int(h) for h in f.readline().strip().split(',')]
        weathers = f.readline().strip().split(',')
        yaws = [float(y) for y in f.readline().strip().split(',')]

    scenario_names = ['{}_{}_{}'.format(h, w, y) for h in hours for w in weathers for y in yaws]
    commands = {key: [] for key in scenario_names}

    for name in scenario_names:
        if not os.path.isdir(os.path.join(save_folder, name, 'IMG')):
            os.makedirs(os.path.join(save_folder, name, 'IMG'))

    with gzip.open(os.path.join(save_folder, file_name), 'rb') as f:
        skip = 1
        frame_id = -1
        length = len(scenario_names)
        while True:
            try:
                message = pickle.load(f)
                if frame_id < skip*length:
                    frame_id += 1
                    continue

                scenario = scenario_names[frame_id%length]

                commands[scenario].append([frame_id//length, message['steering'], message['throttle'], message['brake']])
                frame = frame2numpy(message['frame'], (320,160))

                cv2.imwrite(os.path.join(save_folder, scenario, 'IMG', '%05d.png' % (frame_id//length)), frame)

                if args.show:
                    cv2.imshow('GTAV---{}[{}]'.format(args.file_prefix, scenario), frame)
                    cv2.waitKey(1)

                frame_id += 1
            except Exception as e:
                print(e)
                break

        for k, v in commands.items():
            np.savetxt(os.path.join(save_folder, k, 'commands.csv'), np.asarray(v), delimiter=',')