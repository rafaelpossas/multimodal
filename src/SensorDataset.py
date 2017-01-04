import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from keras.preprocessing import sequence

class SensorDataset():


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
                activity = filename[0:5]
                if activity in activities_files.keys():
                    activities_files[activity].append(filename)
                else:
                    activities_files[activity] = [filename]


        return activities_files  # Self-explanatory.

    def get_dataframes(self, activities_files):
        list_x = []
        list_y = []
        for actvity in activities_files:
            for file in activities_files[actvity]:
                df = pd.read_csv(self.root_dir+"/"+file, index_col=None, header=None)
                sensor = sequence.pad_sequences(df.values.T, maxlen=150, dtype='float32')
                list_x.append(sensor.T)
                list_y.append([self.activity_dict[actvity][0]])
        #print(np.array(list_x).shape)
        #print(np.array(list_y).shape)
        #print(np.mean(list_x))
        #print(np.std(list_x))
        return np.array(list_x), self.one_hot(np.array(list_y))

    def one_hot(self, y_):
        # Function to encode output labels from number indexes
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

        y_ = y_.reshape(len(y_))
        n_values = np.max(y_) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.activity_dict = {
            'act01': (0, 'walking'), 'act02': (1, 'walking upstairs'), 'act03': (2, 'walking downstairs'),
            'act04': (3, 'riding elevator up'), 'act05': (4, 'riding elevator down'), 'act06':(5, 'riding escalator up'),
            'act07': (6, 'riding escalator down'), 'act08': (7, 'sitting'), 'act09': (8, 'eating'), 'act10': (9, 'drinking'),
            'act11': (10, 'texting'), 'act12': (11, 'phone calls'), 'act13': (12, 'working on pc'), 'act14': (13, 'reading'),
            'act15': (14, 'writing sentences'), 'act16': (15, 'organizing files'), 'act17': (16, 'running'), 'act18': (17, 'push-ups'),
            'act19': (18, 'sit-ups'), 'act20': (19, 'cycling')

        }

        self.sensor_cols = ['accx', 'accy', 'accz',
                            'grax', 'gray', 'graz',
                            'gyrx', 'gyry', 'gyrz',
                            'lacx', 'lacy', 'lacz',
                            'magx', 'magy', 'magz',
                            'rotx', 'roty', 'rotz', 'rote']

        act = self.get_filepaths(self.root_dir)
        lst_x, lst_y = self.get_dataframes(act)
        lst_x, lst_y = shuffle(lst_x, lst_y, random_state=0)
        train_size = int(len(lst_x)*0.80)
        test_val_size = int(len(lst_x)*0.10)
        print(lst_x.shape)
        self.x_train, self.x_val, self.x_test = lst_x[0:train_size, :, :], \
                                                lst_x[train_size:(train_size+test_val_size), :, :], \
                                                lst_x[(train_size+test_val_size):, :, :]

        self.y_train, self.y_val, self.y_test = lst_y[0:train_size, :], \
                                                lst_y[train_size:(train_size+test_val_size), :], \
                                                lst_y[(train_size+test_val_size):, :]

    def predict_timeseries(self):
        dt = SensorDataset("/Users/rafaelpossas/Dev/multimodal/sensor")
        scaler = MinMaxScaler()

        X = np.vstack((dt.x_train[:, :, :1], dt.x_test[:, :, :1]))
        X = X.reshape([X.shape[0] * X.shape[1], 1])
        X = scaler.fit_transform(X)

        y = np.vstack((dt.y_train, dt.y_test))

        accX_train = dt.x_train[:, :, :1]
        accX_train = accX_train.reshape([accX_train.shape[0] * accX_train.shape[1], 1])
        accX_test = dt.x_test[:, :, :1]
        accX_test = accX_test.reshape([accX_test.shape[0] * accX_test.shape[1], 1])
        accY_train = dt.y_train
        accY_test = dt.y_test

        # plt.plot(accX)
        # plt.show()
        accX_train = accX_train.ravel(order='C')
        accX_test = accX_test.ravel(order='C')

        accX_train = scaler.fit_transform(accX_train[:, np.newaxis])
        accX_test = scaler.fit_transform(accX_test[:, np.newaxis])

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        accX_train, accY_train = create_dataset(accX_train)
        accX_test, accY_test = create_dataset(accX_test)
        accX_train = np.reshape(accX_train, (accX_train.shape[0], 1, accX_train.shape[1]))
        accX_test = np.reshape(accX_test, (accX_test.shape[0], 1, accX_test.shape[1]))
        print(accX_train.shape, accX_test.shape)

        model = Sequential()
        model.add(LSTM(4, input_dim=1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(accX_train, accY_train, nb_epoch=100, batch_size=100, verbose=2)

        trainPredict = model.predict(accX_train)
        testPredict = model.predict(accX_test)

        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([accY_train])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([accY_test])
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(X)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[1:len(trainPredict) + 1, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(X)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (1 * 2) + 1:len(X) - 1, :] = testPredict
        # plot baseline and predictions
        plt.plot(scaler.inverse_transform(X))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()


if __name__=='__main__':
    dt = SensorDataset("/Users/rafaelpossas/Dev/multimodal/sensor")
    scaler = MinMaxScaler()
    num_sensor = 9

    accX_train = dt.x_train[:, :, : num_sensor]
    accX_val = dt.x_val[:, :, :num_sensor]
    accX_test = dt.x_test[:, :, : num_sensor]
    #accX_train = accX_train.reshape([accX_train.shape[0] * accX_train.shape[1], 1])
    #accX_test = accX_test.reshape([accX_test.shape[0] * accX_test.shape[1], 1])
    accY_train = dt.y_train
    accY_val = dt.y_val
    accY_test = dt.y_test

    # plt.plot(accX)
    # plt.show()
    #accX_train = accX_train.ravel(order='C')
    #accX_test = accX_test.ravel(order='C')

    #accX_train = scaler.fit_transform(accX_train[:, np.newaxis])
    #accX_test = scaler.fit_transform(accX_test[:, np.newaxis])
    split_train_x = []
    split_train_y = []
    split_val_x = []
    split_val_y = []
    split_test_x = []
    split_test_y = []


    n = 50 # group size
    m = 0 # overlap size
    num_splits = 5

    for i in range(accX_train.shape[0]):
        currentsplit = [accX_train[i, j:j+n-m] for j in xrange(0,len(accX_train[i]), n-m)]
        split_train_x = split_train_x + currentsplit
        split_train_y = split_train_y + [accY_train[i] for x in range(len(currentsplit))]

    for i in range(accX_test.shape[0]):
        currentsplit = [accX_val[i, j:j+n-m] for j in xrange(0,len(accX_val[i]), n-m)]
        split_val_x = split_val_x + currentsplit
        split_val_y = split_val_y + [accY_val[i] for x in range(len(currentsplit))]

    for i in range(accX_test.shape[0]):
        currentsplit = [accX_test[i, j:j+n-m] for j in xrange(0,len(accX_test[i]), n-m)]
        split_test_x = split_test_x + currentsplit
        split_test_y = split_test_y + [accY_test[i] for x in range(len(currentsplit))]


    split_train_x = np.array(split_train_x)
    split_train_y = np.array(split_train_y)
    split_val_x = np.array(split_val_x)
    split_val_y = np.array(split_val_y)
    split_test_x = np.array(split_test_x)
    split_test_y = np.array(split_test_y)

    model = Sequential()
    model.add(LSTM(128, input_shape=(split_train_x.shape[1], split_train_x.shape[2])))
    model.add(Dense(split_train_y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(split_train_x, split_train_y, validation_data=(split_val_x, split_val_y), nb_epoch=100, batch_size=16)
    scores = model.evaluate(split_test_x, split_test_y, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print(model.summary())