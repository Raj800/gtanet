from deepgtav.messages import Start, Stop, Config, Dataset, frame2numpy, Scenario
from deepgtav.client import Client
import argparse
import cv2
import time
import os


def reset():
    ''' Resets position of car to a specific location '''
    # Same conditions as below |
    dataset = Dataset(rate=60, frame=[800, 600], throttle=True, brake=True, steering=True, location=True,)# drivingMode=True)
    scenario = Scenario(weathers='EXTRASUNNY', vehicle='blista', times=[12, 0])#, drivingMode=[786603, 40.0])
    # , location=[-2573.13916015625, 3292.256103515625, 13.241103172302246])
    client.sendMessage(Config(scenario=scenario, dataset=dataset))


# Stores a pickled dataset file with data coming from DeepGTAV
if __name__ == '__main__':
    dataset_id = 1
    filename = 'dataset_minimap_800x600-{}'.format(dataset_id)
    np_filename = 'dataset_minimap_800x600-{}.pz'.format(dataset_id)
    while os.path.exists(np_filename):
        print('{} already present. Moving along,'.format(np_filename))
        dataset_id += 1
        np_filename = 'dataset_minimap_800x600-{}.pz'.format(dataset_id)
    while True:
        parser = argparse.ArgumentParser(description=None)
        parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
        parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
        parser.add_argument('-d', '--dataset_path', default=np_filename, help='Place to store the dataset')
        args = parser.parse_args()
        # Creates a new connection to DeepGTAV using the specified ip and port
        print(np_filename)
        client = Client(ip=args.host, port=args.port, datasetPath=np_filename, compressionLevel=9)
        # Dataset options
        dataset = Dataset(rate=60, frame=[800, 600], throttle=True, brake=True, steering=True, location=True)#, drivingMode=True)
        # Automatic driving scenario
        scenario = Scenario(weathers='EXTRASUNNY', vehicle='blista', times=[12, 0])#, drivingMode=[6, 30.0])
        # ,location=[-2573.13916015625, 3292.256103515625, 13.241103172302246])

        # FLAGS - CONVERTED
        # INTEGER - NAME / DESC
        # OF
        # THE
        # DRIVING
        # STYLE
        # 01000000000011000000000000100101 - 1074528293 - "Rushed"
        # 00000000000011000000000000100100 - 786468 - "Avoid Traffic"
        # 00000000000000000000000000000110 - 6 - "Avoid Traffic Extremely"
        # 00000000000011000000000010101011 - 786603 - "Normal"
        # 00000000001011000000000000100101 - 2883621 - "Ignore Lights"
        # 00000000000000000000000000000101 - 5 - "Sometimes Overtake Traffic"

        i = 10
        while i:
            print('Starting Data collection in', i, 'sec')
            i -= 1
            time.sleep(1)

        client.sendMessage(Start(scenario=scenario, dataset=dataset))  # Start request

        count = 0
        old_location = [0, 0, 0]

        while True:  # Main loop
            try:
                # Message recieved as a Python dictionary
                message = client.recvMessage()
                # frame_img = frame2numpy(message['frame'],[320, 160])
                if (count % 100) == 0:
                    print(count)
                # Checks if car is stuck, resets position if it is
                if (count % 250) == 0:
                    new_location = message['location']
                    # Float position converted to ints so it doesn't have to be in the exact same position to be reset
                    if int(new_location[0]) == int(old_location[0]) and int(new_location[1]) == int(
                            old_location[1]) and int(new_location[2]) == int(old_location[2]):
                        reset()
                    old_location = message['location']
                    print('At location: ' + str(old_location))
                count += 1
                #if count >= 10000:
                 #   dataset_id += 1
                  #  break

            except KeyboardInterrupt:
                i = input('Paused. Press p to continue and e to exit... ')
                if i == 'p':
                    continue
                elif i == 'e':
                    break

        # DeepGTAV stop message
        client.sendMessage(Stop())
        client.close()

