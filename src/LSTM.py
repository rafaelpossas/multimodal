import numpy as np


from src.SensorDataset import SensorDataset
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model

from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from itertools import *
from keras.layers import Dropout, Input, RepeatVector
from src.Utils import *


class RegressionLSTM():

    def __init__(self, scaler):
        self.scaler = scaler

    def format_data(self, dt):
        x_train = dt.x_train[:, :, :]
        axis = x_train.shape[2]
        x_train = x_train.reshape([x_train.shape[0] * x_train.shape[1], axis])
        x_test = dt.x_test[:, :, :]
        x_test = x_test.reshape([x_test.shape[0] * x_test.shape[1], axis])
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

    def fit_transform(self, model, x_train, y_train, x_test, nb_epoch=100, batch_size=100, verbose=2):
        model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)
        trainPredict = model.predict(x_train)
        testPredict = model.predict(x_test)

        return trainPredict, testPredict




class SensorLSTM():

    def get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = Sequential()
        model.add(LSTM(layer_size, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def fit_transform(self, model, dataset, nb_epoch=100, batch_size=16, callbacks=[],verbose=0):
        model.fit(dataset.x_train, dataset.y_train, validation_split=0.1,
                  nb_epoch=nb_epoch, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        scores = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
        return scores




if __name__=='__main__':
    dt = SensorDataset("/home/rafaelpossas/dev/multimodal_dataset/sensor")
    scaler = MinMaxScaler()
    lstm = RegressionLSTM(scaler)
    dt.load_dataset(train_size=0.9, split_train=True, group_size=75, step_size=75,
                    selected_sensors=['accx'])
    #x_train, x_test, y_train, y_test = lstm.format_data(dt)
    # train_prediction, test_prediction = lstm.fit_transform(lstm.get_model(), x_train, y_train, x_test,
    #                                                        nb_epoch=10,
    #                                                        batch_size=100,
    #                                                        verbose=0)
    # plot_predictions([dt], [train_prediction], [test_prediction],
    #                         [y_train], [y_test], ['accx'], scaler)


    latent_dim = 25
    input_dim = 1
    timesteps = 75
    dropout_rate = 0.3
    nb_epoch = 30
    batch_size = 10
    verbose = 1

    # x_train = dt.x_train.ravel(order='C')
    # x_train = scaler.fit_transform(x_train[:, np.newaxis])
    # x_train = x_train.reshape(dt.x_train.shape[0], timesteps, input_dim)
    #
    # x_test = dt.x_test.ravel(order='C')
    # x_test = scaler.fit_transform(x_test[:, np.newaxis])
    # x_test = x_test.reshape(dt.x_test.shape[0], timesteps, input_dim)
    #
    #
    # mid_index = x_train.shape[0]/2
    #
    # x_train_supervised = x_train[0:int(mid_index), :, :]
    # y_train_supervised = dt.y_train[0:int(mid_index)]
    #
    # x_train_unsupervised = x_train[int(mid_index):]

    model = Sequential()
    model.add(LSTM(output_dim=latent_dim, input_shape=(timesteps, input_dim)))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(input_dim, return_sequences=True))
    model.compile(optimizer='rmsprop', loss="mse")
    model.fit(dt.x_train, dt.x_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)

    inputs = Input(shape=(timesteps, input_dim))
    encoder = Model(inputs, model.layers[0](inputs))

    encoded_input = Input(shape=(timesteps, latent_dim))
    decoder = Model(input=encoded_input, output=model.layers[-1](encoded_input))

    #encoder_weights = model.layers[0].get_weights()

    # sensor_model = Sequential()
    # sensor_model.add(LSTM(latent_dim, input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]), weights=encoder_weights))
    # sensor_model.add(Dropout(0.3))
    # sensor_model.add(Dense(dt.y_train.shape[1], activation='softmax'))
    # sensor_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # sensor_model.fit(dt.x_test, dt.y_test, nb_epoch=25, batch_size=20, verbose=2)

    sample = dt.x_test[0, :, :]
    sample = sample.reshape(1, sample.shape[0], sample.shape[1])
    code = encoder.predict(sample)
    code = np.repeat(code, timesteps)
    code = code.reshape(1, timesteps, latent_dim)
    reconstructed = decoder.predict(code)

    plt.plot(sample[0])
    plt.plot(reconstructed[0])


    # inputs = Input(shape=(timesteps, input_dim))
    # encoded = LSTM(latent_dim)(inputs)
    #
    # decoded = RepeatVector(timesteps)(encoded)
    # decoded = LSTM(input_dim, return_sequences=True)(decoded)
    #
    # sequence_autoencoder = Model(inputs, decoded)
    # encoder = Model(inputs, encoded)
    #
    # encoded_input = Input(shape=(timesteps,latent_dim))
    # layer = sequence_autoencoder.layers[-1]
    # decoder = Model(input=encoded_input, output=layer(encoded_input))
    #
    # sequence_autoencoder.compile(optimizer='adadelta', loss="mean_squared_error")
    # sequence_autoencoder.fit(dt.x_train, dt.x_train, nb_epoch=20, batch_size=20)
    #
    # sample = dt.x_test[0, :, :]
    # encoded_seq = encoder.predict(sample.reshape(1, sample.shape[0], sample.shape[1]))
    # encoded_seq = np.repeat(encoded_seq, timesteps)
    # decoded_seq = decoder.predict(encoded_seq.reshape(1, timesteps, latent_dim))


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
