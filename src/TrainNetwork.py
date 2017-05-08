import numpy as np


from src.SensorDataset_UCI import SensorDatasetUCI
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, RepeatVector
from src.Utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
import LSTM

latent_dim = 100
input_dim = 3
timesteps = 150
step_size = 150
nb_epoch = 10
batch_size = 6000
verbose = 1


dt = SensorDatasetUCI("/Users/rafaelpossas/Dev/multimodal/uci_cleaned")
# scaler = MinMaxScaler()
# lstm = RegressionLSTM(scaler)
dt.load_dataset(train_size=0.9, group_size=timesteps, step_size=step_size)
lstm_network = LSTM.SensorLSTM()
model = lstm_network.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]), output_shape=dt.y_train.shape[1],
                               layer_size=150, optimizer='rmsprop', dropout=0.2)
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=30)


#Scores
scores = lstm_network.fit_transform(model, dt, epochs=1000, callbacks=[early_stopping])
acc = (scores[1] * 100)
print("Accuracy: %.2f%%" % acc)