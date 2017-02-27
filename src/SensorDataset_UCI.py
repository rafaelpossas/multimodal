import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from keras.preprocessing import sequence
import chardet

from keras.preprocessing import sequence

class SensorDatasetUCI():

    def __init__(self, root_dir):
        self.rows_limit = 1500000
        self.root_dir = root_dir
        self.lst_x = []
        self.lst_y = []
        self.activity_dict = {
            'walk': 0, 'sit': 1, 'bike': 2, 'stairs': 3,'stand': 4

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
                if(filename !=".DS_Store"):
                    file_paths.append(filename)  # Add it to the list.
                    activity = filename.split("_")[1]
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

    def _load_from_file(self, activities_files, selected_sensors, chunk_size=150):
        list_x = []
        list_y = []
        sizes = []
        for actvity in activities_files:
            for file in activities_files[actvity]:
                # with open(self.root_dir+"/"+actvity+"/"+file, 'rb') as f:
                #     result = chardet.detect(f.read())  # or readline if the file is large
                df = pd.read_csv(self.root_dir+"/"+actvity+"/"+file, sep='\t', index_col=None, encoding="UTF-16LE", engine="python")
                df = df[selected_sensors] if len(selected_sensors) > 0 else df
                sizes.append(len(df.values))
                def greedy_split(arr, n, axis=0):
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

                    # compute the size of each of the first n-1 blocks
                    block_size = np.ceil(length / float(n))

                    # the indices at which the splits will occur
                    ix = np.arange(block_size, length, block_size).astype(int)

                    return np.split(arr, ix, axis)
                #values = np.array_split(df.values, (len(df.values)//chunk_size)+1)
                values = greedy_split(df.values, (len(df.values)//chunk_size)+1)
                for vl in values:

                    while len(vl) < chunk_size:
                        dims = []
                        for index in range(0, len(selected_sensors)):
                            dims.append(np.mean(vl[:, index]))
                        dims = np.reshape(dims, (1, len(selected_sensors)))
                        vl = np.vstack((vl, dims))
                    list_x.append(vl)
                    list_y.append([self.activity_dict[actvity]])
        list_x = np.reshape(list_x, (len(list_x), chunk_size, len(selected_sensors)))
        list_y = self.one_hot(np.array(list_y))
        return list_x, list_y

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

# if __name__=='__main__':
#     dt = SensorDatasetUCI("/Users/rafaelpossas/Dev/multimodal/uci_cleaned")
#     dt.load_dataset(train_size=0.9, split_train=True, group_size=75, step_size=75,
#                     selected_sensors=['X'])