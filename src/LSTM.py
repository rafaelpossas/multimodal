import numpy as np
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Dropout, RepeatVector
from keras.optimizers import RMSprop
from src.Utils import *
from src.MultimodalDataset import MultimodalDataset

from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau


class RegressionLSTM:

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

class SensorLSTM:

    def train_model(self, train_file=None, test_file=None, chunk_size=5, step_size=1):


        # scaler = MinMaxScaler()
        train_file = h5py.File(train_file)
        test_file = h5py.File(test_file)
        dataset = MultimodalDataset()

        train_sns_x = train_file['x_sns'][:]
        train_sns_y = train_file['y'][:]
        train_onehot_y = np.eye(20)[train_sns_y.astype(int)]

        train_sns_x, train_onehot_y = dataset.split_windows(chunk_size, step_size, train_sns_x, train_onehot_y)

        print(train_sns_x.shape)

        test_sns_x = test_file['x_sns'][:]
        test_sns_y = test_file['y'][:]
        test_onehot_y = np.eye(20)[test_sns_y.astype(int)]

        test_sns_x, test_onehot_y = dataset.split_windows(chunk_size, step_size, test_sns_x, test_onehot_y)


        print(test_sns_x.shape)

        sensor_lstm = SensorLSTM()
        earlystopping = EarlyStopping(patience=20)
        model_checkpoint = ModelCheckpoint('sensor_model.hdf5', save_best_only=True, monitor='val_acc')
        #reduce_lr = ReduceLROnPlateau(patience=3, min_lr=0.000001, verbose=1)
        rmsprop = RMSprop()
        model_lstm = sensor_lstm.get_model(input_shape=(train_sns_x.shape[1], train_sns_x.shape[2]),
                                           output_shape=train_onehot_y.shape[1], dropout=0.4, layer_size=20,
                                           optimizer='rmsprop')
        #model_lstm.load_weights('sensor_model.hdf5')
        model_lstm.fit(train_sns_x, train_onehot_y, batch_size=1000, epochs=1000,
                           validation_data=(test_sns_x, test_onehot_y), callbacks=[earlystopping, model_checkpoint], verbose=2)
        model_lstm.load_weights("sensor_model.hdf5")
        print(model_lstm.evaluate(test_sns_x, test_onehot_y))
        return model_lstm

    def get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = Sequential()
        model.add(LSTM(layer_size, input_shape=input_shape,return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def fit_transform(self, model, dataset, epochs=100, batch_size=2048, callbacks=[], verbose=1):
        model.fit(dataset.x_train, dataset.y_train, validation_split=0.1,
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        scores = model.evaluate(dataset.x_test, dataset.y_test, verbose=verbose)
        return scores


class AutoencoderLSTM:

    def __init__(self, latent_dim, input_dim, timesteps):
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.latent_dim_2 = int(latent_dim / 2)
        self.latent_dim_3 = int(self.latent_dim_2 / 2)

    def get_stacked_autoencoder_model(self):

        autoencoder_0 = Sequential()
        autoencoder_0.add(LSTM(output_dim=self.latent_dim, input_shape=(self.timesteps, self.input_dim),
                               return_sequences=True))
        autoencoder_0.add(LSTM(output_dim=self.latent_dim_2, return_sequences=True))
        autoencoder_0.add(LSTM(output_dim=self.latent_dim_3))
        autoencoder_0.add(RepeatVector(self.timesteps))
        autoencoder_0.add(LSTM(output_dim=self.latent_dim_2, return_sequences=True))
        autoencoder_0.add(LSTM(output_dim=self.latent_dim, return_sequences=True))
        autoencoder_0.add(LSTM(output_dim=self.input_dim, input_dim=self.latent_dim, return_sequences=True))
        autoencoder_0.output_reconstruction = True

        ae_model_0 = Sequential()
        ae_model_0.add(autoencoder_0)
        ae_model_0.compile(optimizer='adam', loss="mse")

        return ae_model_0

    def get_stacked_model(self, pre_trained_model=None, classes=5):
        sensor_model = Sequential()

        if pre_trained_model is None:
            print("Using Random Weights")
            sensor_model.add(LSTM(output_dim=self.latent_dim, input_shape=(self.timesteps, self.input_dim),
                                  return_sequences=True))

            sensor_model.add(LSTM(output_dim=self.latent_dim_2, return_sequences=True))

            sensor_model.add(LSTM(output_dim=self.latent_dim_3))
        else:
            print("Using AE Pre-Initialized Weights")

            sensor_model.add(LSTM(output_dim=self.latent_dim, input_shape=(self.timesteps, self.input_dim),
                                     weights=pre_trained_model.layers[0].layers[0].get_weights(),
                                     return_sequences=True))

            sensor_model.add(LSTM(output_dim=self.latent_dim_2,
                                  weights=pre_trained_model.layers[0].layers[1].get_weights(),
                                  return_sequences=True))

            sensor_model.add(LSTM(output_dim=self.latent_dim_3,
                                  weights=pre_trained_model.layers[0].layers[2].get_weights()))



        sensor_model.add(Dense(classes, input_dim=int(self.latent_dim / 2), activation='softmax', init='zero'))
        sensor_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return sensor_model

    def fit_ae(self, x_train, model, nb_epoch=5, batch_size=2000,
               verbose=1, save_model=True,filepath="model.hdf5", callbacks=[]):

        model.fit(x_train, x_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose, callbacks=callbacks)
        if save_model:
            model.save_weights(filepath=filepath)


if __name__=='__main__':
    sns_lst = SensorLSTM()
    sns_lst.train_model(train_file='data/multimodal_full_train.hdf5',
                        test_file="data/multimodal_full_test.hdf5", chunk_size=50, step_size=1)