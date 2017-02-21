import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

from keras.preprocessing import sequence

class SensorDatasetUCI():

    def __init__(self, root_dir):
        self.rows_limit = 1500000
        self.root_dir = root_dir
        self.lst_x = []
        self.lst_y = []
        self.activity_dict = {
            'walk': 0, 'sit': 1, 'bike': 2, 'stairsup': 3,
            'stairsdown': 4, 'stand': 5

        }

        self.sensor_columns = ['x', 'y', 'z']



    def get_filepaths(self, directory):
        """
        This function will generate the file names in a directory
        tree by walking the tree either top-down or bottom-up. For each
        directory in the tree rooted at directory top (including top itself),
        it yields a 3-tuple (dirpath, dirnames, filenames).
        """
        file_paths = []  # List which will store all of the full filepaths.
        activities_files = {}
        # Walk the tree.
        for root, directories, files in os.walk(directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                #filepath = os.path.join(root, filename)
                file_paths.append(filename)  # Add it to the list.
                activity = filename.split("_")[0]
                if activity != '.DS':
                    if activity in activities_files.keys():
                        activities_files[activity].append(filename)
                    else:
                        activities_files[activity] = [filename]


        return activities_files  # Self-explanatory.

    def load_dataset(self, train_size=0.8, split_train=True,
                  group_size=50, step_size=0, selected_sensors=[]):

        act = self.get_filepaths(self.root_dir)
        self.lst_x, self.lst_y = self._load_from_file(act, selected_sensors)
        self.lst_x, self.lst_y = shuffle(self.lst_x, self.lst_y, random_state=0)

        train_size = int(len(self.lst_x)*train_size)

        self.x_train, self.x_test = self.lst_x[0:train_size, :, :], \
                                    self.lst_x[train_size:, :, :]

        self.y_train, self.y_test = self.lst_y[0:train_size, :], \
                                    self.lst_y[train_size:, :]
        if split_train:
            self.x_train, self.y_train = self.split_windows(group_size=group_size,
                                                            step_size=step_size, X=self.x_train, y=self.y_train)
            self.x_test, self.y_test = self.split_windows(group_size=group_size,
                                                          step_size=step_size, X=self.x_test, y=self.y_test)


        print("Train {}.{}".format(self.x_train.shape, self.y_train.shape))
        print("Test {}.{}".format(self.x_test.shape, self.y_test.shape))

    def _load_from_file(self, activities_files, selected_sensors):
        list_x = []
        list_y = []
        sizes = []
        for actvity in activities_files:
            for file in activities_files[actvity]:
                df = pd.read_csv(self.root_dir+"/"+file, index_col=None)
                df = df[selected_sensors] if len(selected_sensors) > 0 else df
                values = df.values[0:self.rows_limit]
                sizes.append(len(values))
                values = values.reshape((int(self.rows_limit/150), 150, 1))
                [list_x.append(vl) for vl in values]
                [list_y.append([self.activity_dict[actvity]]) for _ in values]

        print(min(sizes))
        return np.array(list_x), self.one_hot(np.array(list_y))

    def one_hot(self, y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

    def split_windows(self, group_size, step_size, X, y):
        number_groups = int(((X.shape[1]-group_size)/step_size)+1)
        split_xy = [(X[j, i:i + group_size], y[j]) for j in range(len(X)) for i in range(0, number_groups * step_size, step_size)]
        split_x = np.array([x[0] for x in split_xy])
        split_y = np.array([y[1] for y in split_xy])
        return split_x, split_y

if __name__=='__main__':
    dt = SensorDatasetUCI("/Users/rafaelpossas/Dev/multimodal/uci")
    dt.load_dataset(train_size=0.9, split_train=True, group_size=75, step_size=75,
                    selected_sensors=['x'])