import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from keras.preprocessing import sequence
import chardet

from keras.preprocessing import sequence

class SensorDatasetUCI():

    def __init__(self, root_dir, sensors=None, users=None, devices=None, activities=None, axis=None):
        self.rows_limit = 1500000
        self.root_dir = root_dir
        self.lst_x = []
        self.lst_y = []
        self.activity_dict = {'walk': 0, 'sit': 1, 'bike': 2, 'stairs': 3, 'stand': 4}

        self.sensors = ['acc', 'gyr'] if sensors is None else sensors
        self.users = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] if users is None else users
        self.devices = ['nexus4-1', 'nexus4-2', 's3-1', 's3-2', 's3mini-1', 's3mini-2'] if devices is None else devices
        self.acitivies = ['walk', 'sit', 'bike', 'stairs', 'stand'] if activities is None else activities
        self.axis = ['X', 'Y', 'Z'] if axis is None else axis

        self.sensor_columns = ['x', 'y', 'z']

    def get_filepaths(self, directory):
        """
        This function will generate the file names in a directory
        tree by walking the tree either top-down or bottom-up. For each
        directory in the tree rooted at directory top (including top itself),
        it yields a 3-tuple (dirpath, dirnames, filenames).
        """
        activities_files = {}
        for root, directories, files in os.walk(directory):
            if len(directories) == 0:
                activity = root.split("/")[-1]
                for user in self.users:
                    for device in self.devices:
                        files_array = []

                        for sensor in self.sensors:
                            file_name = str(root + "/" + sensor + "_" + activity + "_" + user + "_" + device + ".csv")
                            if os.path.isfile(file_name):
                                files_array.append(file_name)

                        if len(files_array) == len(self.sensors):
                            if activity in activities_files.keys():
                                activities_files[activity].append(files_array)
                            else:
                                activities_files[activity] = {}
                                activities_files[activity] = [files_array]

        return activities_files

    def load_dataset(self, train_size=0.8, group_size=50, step_size=0):

        act = self.get_filepaths(self.root_dir)
        self.lst_x, self.lst_y = self._load_from_file(act, chunk_size=group_size, step_size=step_size)
        self.lst_x, self.lst_y = shuffle(self.lst_x, self.lst_y, random_state=0)

        train_size = int(len(self.lst_x)*train_size)

        self.x_train, self.x_test = self.lst_x[0:train_size, :, :], \
                                    self.lst_x[train_size:, :, :]

        self.y_train, self.y_test = self.lst_y[0:train_size, :], \
                                    self.lst_y[train_size:, :]


        print("Train {}.{}".format(self.x_train.shape, self.y_train.shape))
        print("Test {}.{}".format(self.x_test.shape, self.y_test.shape))

    def _load_from_file(self, activities_files, chunk_size=150, step_size=150):
        def greedy_split(arr, axis=0):
            """Greedily splits an array into n blocks.

            Splits array arr along axis into n blocks such that:
                - blocks 1 through n-1 are all the same size
                - the sum of all block sizes is equal to arr.shape[axis]
                - the last block is nonempty, and not bigger than the other blocks

            Intuitively, this "greedily" splits the array along the axis by making
            the first blocks as big as possible, then putting the leftovers in the
            last block.
            """
            length = arr.shape[axis]

            # the indices at which the splits will occur
            ix = np.arange(0, length, step_size).astype(int)

            return [arr[i:i + int(chunk_size), :] for i in ix]
        list_x = []
        list_y = []
        sizes = []
        for actvity in activities_files:
            for file_array in activities_files[actvity]:
                # with open(self.root_dir+"/"+actvity+"/"+file, 'rb') as f:
                #     result = chardet.detect(f.read())  # or readline if the file is large
                all_sensors_x = []
                df = None
                print("Loading: "+str(file_array))
                for file in file_array:
                    if df is None:
                        df = pd.read_csv(file, sep='\t', index_col=None, encoding="UTF-16LE")
                        df = df[self.axis] if len(self.axis) > 0 else df
                        df.columns = range(df.shape[1])
                    else:
                        aux = pd.read_csv(file, sep='\t', index_col=None, encoding="UTF-16LE")
                        aux = aux[self.axis] if len(self.axis) > 0 else aux
                        aux.columns = range(df.shape[1], df.shape[1]+aux.shape[1])
                        df = pd.concat([df, aux], axis=1)
                df.dropna(inplace=True)
                sizes.append(len(df.values))

                #values = np.array_split(df.values, (len(df.values)//chunk_size)+1)
                values = greedy_split(df.values)
                for vl in values:

                    while len(vl) < chunk_size:
                        dims = []
                        for index in range(0, len(self.axis)*len(self.sensors)):
                            dims.append(np.mean(vl[:, index]))
                        dims = np.reshape(dims, (1, len(self.axis) * len(self.sensors)))
                        vl = np.vstack((vl, dims))
                    list_x.append(vl)
                    list_y.append([self.activity_dict[actvity]])
        list_x = np.asarray(list_x)
        list_y = self.one_hot(np.array(list_y))
        return list_x, list_y

    def one_hot(self, y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


# if __name__=='__main__':
#     dt = SensorDatasetUCI("/Users/rafaelpossas/Dev/multimodal/uci_cleaned")
#     dt.load_dataset(train_size=0.9, split_train=True, group_size=75, step_size=75,
#                     selected_sensors=['X'])