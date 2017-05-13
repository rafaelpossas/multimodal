import numpy as np


from src.SensorDataset_UCI import SensorDatasetUCI
from src.LSTM import SensorLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
lstm_network = SensorLSTM()
model = lstm_network.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]), output_shape=dt.y_train.shape[1],
                               layer_size=150, optimizer='rmsprop', dropout=0.2)
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10)


#Scores
scores = lstm_network.fit_transform(model, dt, epochs=15)
acc = (scores[1] * 100)
print("Accuracy: %.2f%%" % acc)

dt = SensorDatasetUCI("/Users/rafaelpossas/Dev/multimodal/uci_cleaned", sensors=['acc'])
# scaler = MinMaxScaler()
# lstm = RegressionLSTM(scaler)
dt.load_dataset(train_size=0.9, group_size=timesteps, step_size=step_size)
lstm_network = SensorLSTM()
model = lstm_network.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]), output_shape=dt.y_train.shape[1],
                               layer_size=150, optimizer='rmsprop', dropout=0.2)
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10)


#Scores
scores = lstm_network.fit_transform(model, dt, epochs=15)
acc = (scores[1] * 100)
print("Accuracy: %.2f%%" % acc)


dt = SensorDatasetUCI("/Users/rafaelpossas/Dev/multimodal/uci_cleaned", sensors=['gyr'])
# scaler = MinMaxScaler()
# lstm = RegressionLSTM(scaler)
dt.load_dataset(train_size=0.9, group_size=timesteps, step_size=step_size)
lstm_network = SensorLSTM()
model = lstm_network.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]), output_shape=dt.y_train.shape[1],
                               layer_size=150, optimizer='rmsprop', dropout=0.2)
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=10)


#Scores
scores = lstm_network.fit_transform(model, dt, epochs=15)
acc = (scores[1] * 100)
print("Accuracy: %.2f%%" % acc)
