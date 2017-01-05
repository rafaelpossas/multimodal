import numpy as np
import matplotlib.pyplot as plt
import math
from SensorDataset import SensorDataset
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from itertools import *

class SensorLSTM():

    def get_model(self, input_shape, output_shape, layer_size=128):
        model = Sequential()
        model.add(LSTM(layer_size, input_shape=input_shape))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def fit_transform(self, model, dataset, nb_epoch=100, batch_size=16, callbacks=[]):
        model.fit(dataset.x_train, dataset.y_train, validation_data=(dataset.x_val, dataset.y_val),
                  nb_epoch=nb_epoch, batch_size=batch_size, callbacks=callbacks)
        scores = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
        return scores

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
    # lstm = SensorLSTM()
    # scaler = MinMaxScaler()
    # num_sensor = 9
    # #Loading Data and creating model
    # dt.load_dataset(selected_sensors=['accx', 'accy', 'accz'])
    #
    # model = lstm.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]),
    #                        output_shape=dt.y_train.shape[1], layer_size=256)
    # #Callbacks
    # checkpointer = ModelCheckpoint(filepath="./models/weights.hdf5", verbose=0, save_best_only=True)
    # reduce_lr_on_plateau = ReduceLROnPlateau(monitor="loss", factor=0.01)
    # early_stopping = EarlyStopping(monitor="loss", min_delta=0.001, patience=20)
    #
    # #Scores
    # scores = lstm.fit_transform(model, dt, nb_epoch=1000, callbacks=[checkpointer, reduce_lr_on_plateau, early_stopping])
    #
    # print("Accuracy: %.2f%%" % (scores[1] * 100))
    # print(model.summary())
    sensor_columns = [['accx', 'accy', 'accz'],
                      ['grax', 'gray', 'graz'],
                      ['gyrx', 'gyry', 'gyrz'],
                      ['lacx', 'lacy', 'lacz'],
                      ['magx', 'magy', 'magz'],
                      ['rotx', 'roty', 'rotz', 'rote']]
    a = [x for l in range(0, len(sensor_columns)) for x in combinations(sensor_columns, l)]
    print(len(a))