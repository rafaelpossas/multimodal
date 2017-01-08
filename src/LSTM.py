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
from keras.layers import Dropout

class SensorLSTM():

    def get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = Sequential()
        model.add(LSTM(layer_size, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def fit_transform(self, model, dataset, nb_epoch=100, batch_size=16, callbacks=[]):
        model.fit(dataset.x_train, dataset.y_train, validation_split=0.1,
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
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
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
    dt = SensorDataset("/home/rafaelpossas/dev/multimodal_dataset/sensor")
    best_accuracy = 60
    sensor_columns = [['accx', 'accy', 'accz'],
                      ['grax', 'gray', 'graz'],
                      ['gyrx', 'gyry', 'gyrz'],
                      ['lacx', 'lacy', 'lacz'],
                      ['magx', 'magy', 'magz'],
                      ['rotx', 'roty', 'rotz', 'rote']]
    sensors = [x for l in range(1, len(sensor_columns)) for x in combinations(sensor_columns, l)]
    grid = dict(optimizers=['rmsprop', 'adagrad', 'adam','adadelta'],
                layer_size=['32', '64', '128', '256'],
                group_size=['10', '30', '50', '75'],
                dropout=['0.2', '0.4', '0.6', '0.8'])
    grid_comb = [(x, y, z, w) for x in grid['optimizers'] for y in grid['layer_size'] for z in grid['group_size'] for w in grid['dropout']]
    lstm = SensorLSTM()
    scaler = MinMaxScaler()
    for sensor in sensors:
        sensor = [e for l in sensor for e in l]
        #Loading Data and creating model
        for grd in grid_comb:
            print("Current Sensors {}".format(sensor))
            dt.load_dataset(selected_sensors=sensor,
                            group_size=int(grd[2]), step_size=int(grd[2]), train_size=0.9)

            model = lstm.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]),
                                   output_shape=dt.y_train.shape[1], layer_size=int(grd[1]),
                                   optimizer=grd[0], dropout=float(grd[3]))
            #Callbacks
            #filepath = "./models/{val_acc:.2f}_"+'_'.join(sensor)+".hdf5"
            #checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)
            #reduce_lr_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.01, verbose=1)
            early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=30)


            #Scores
            scores = lstm.fit_transform(model, dt, nb_epoch=1000, callbacks=[early_stopping])
            acc = (scores[1] * 100)
            print("Accuracy: %.2f%%" % acc)
            filepath = "./models/%.2f_" % acc + '_'.join(sensor)+'_'+'_'.join(grd) + ".hdf5"
            if acc >= best_accuracy:
                best_accuracy = acc
                model.save_weights(filepath=filepath)
