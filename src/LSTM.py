import numpy as np
import matplotlib.pyplot as plt
import math
from SensorDataset import SensorDataset
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from itertools import *
from keras.layers import Dropout


class RegressionLSTM():

    def __init__(self):
        self.scaler = MinMaxScaler()

    def format_data(self, dt):
        x_train = dt.x_train[:, :, :]
        x_train = x_train.reshape([x_train.shape[0] * x_train.shape[1], 1])
        x_test = dt.x_test[:, :, :]
        x_test = x_test.reshape([x_test.shape[0] * x_test.shape[1], 1])
        y_train = dt.y_train
        y_test = dt.y_test

        # plt.plot(accX)
        # plt.show()
        x_train = x_train.ravel(order='C')
        x_test = x_test.ravel(order='C')

        x_train = self.scaler.fit_transform(x_train[:, np.newaxis])
        x_test = self.scaler.fit_transform(x_test[:, np.newaxis])

        # convert an array of values into a dataset matrix
        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        x_train, y_train = create_dataset(x_train)
        x_test, y_test = create_dataset(x_test)
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
        return x_train, x_test, y_train, y_test

    def get_model(self):
        model = Sequential()
        model.add(LSTM(4, input_dim=1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

    def fit_transform(self, model, x_train, y_train, x_test, nb_epoch=100, batch_size=100,verbose=2):
        model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2)
        trainPredict = model.predict(x_train)
        testPredict = model.predict(x_test)

        return trainPredict, testPredict

    def plot_predictions(self, dt, train_prediction, test_prediction, y_train, y_test):

        X = np.vstack((dt.x_train[:, :, :], dt.x_test[:, :, :]))
        X = X.reshape([X.shape[0] * X.shape[1], 1])
        X = self.scaler.fit_transform(X)


        train_prediction = self.scaler.inverse_transform(train_prediction)
        y_train = self.scaler.inverse_transform([y_train])

        test_prediction = self.scaler.inverse_transform(test_prediction)
        y_test = self.scaler.inverse_transform([y_test])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[0], train_prediction[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[0], test_prediction[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(X)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[1:len(train_prediction) + 1, :] = train_prediction
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(X)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_prediction) + (1 * 2) + 1:len(X) - 1, :] = test_prediction
        # plot baseline and predictions
        plt.plot(self.scaler.inverse_transform(X))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        return plt



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




if __name__=='__main__':
    dt = SensorDataset("/Users/rafaelpossas/Dev/multimodal/sensor")
    lstm = RegressionLSTM()
    dt.load_dataset(train_size=0.8, split_train=True, group_size=150, step_size=150, selected_sensors=['accx'])
    x_train, x_test, y_train, y_test = lstm.format_data(dt)
    train_prediction, test_prediction = lstm.fit_transform(lstm.get_model(), x_train, y_train, x_test,
                                                           nb_epoch=10,
                                                           batch_size=100,
                                                           verbose=0)
    lstm.plot_predictions(dt, train_prediction, test_prediction,
                            y_train, y_test)
    # best_accuracy = 60
    # sensor_columns = [['accx', 'accy', 'accz'],
    #                   ['grax', 'gray', 'graz'],
    #                   ['gyrx', 'gyry', 'gyrz'],
    #                   ['lacx', 'lacy', 'lacz'],
    #                   ['magx', 'magy', 'magz'],
    #                   ['rotx', 'roty', 'rotz', 'rote']]
    # sensors = [x for l in range(1, len(sensor_columns)) for x in combinations(sensor_columns, l)]
    # # grid = dict(optimizers=['rmsprop', 'adagrad', 'adam','adadelta'],
    # #             layer_size=['32', '64', '128', '256'],
    # #             group_size=['10', '30', '50', '75'],
    # #             dropout=['0.2', '0.4', '0.6', '0.8'])
    # grid = dict(optimizers=['rmsprop'],
    #             layer_size=['64'],
    #             group_size=['75'],
    #             dropout=['0.4'])
    # grid_comb = [(x, y, z, w) for x in grid['optimizers'] for y in grid['layer_size'] for z in grid['group_size'] for w in grid['dropout']]
    # lstm = SensorLSTM()
    # scaler = MinMaxScaler()
    # for sensor in sensors:
    #     sensor = [e for l in sensor for e in l]
    #     #Loading Data and creating model
    #     for grd in grid_comb:
    #         print("Current Sensors {}".format(sensor))
    #         dt.load_dataset(selected_sensors=sensor,
    #                         group_size=int(grd[2]), step_size=int(grd[2]), train_size=0.9)
    #
    #         model = lstm.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]),
    #                                output_shape=dt.y_train.shape[1], layer_size=int(grd[1]),
    #                                optimizer=grd[0], dropout=float(grd[3]))
    #         #Callbacks
    #         #filepath = "./models/{val_acc:.2f}_"+'_'.join(sensor)+".hdf5"
    #         #checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)
    #         #reduce_lr_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.01, verbose=1)
    #         early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=30)
    #
    #
    #         #Scores
    #         scores = lstm.fit_transform(model, dt, nb_epoch=1000, callbacks=[early_stopping])
    #         acc = (scores[1] * 100)
    #         print("Accuracy: %.2f%%" % acc)
    #         filepath = "./models/%.2f_" % acc + '_'.join(sensor)+'_'+'_'.join(grd) + ".hdf5"
    #         #if acc >= best_accuracy:
    #         #best_accuracy = acc
    #         model.save_weights(filepath=filepath)
