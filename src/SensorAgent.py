import argparse
from keras.callbacks import *
import keras
from MultimodalDataset import MultimodalDataset


class SensorAgent(object):
    model = None

    def __init__(self):
        self.dataset = None

    def predict(self, input=None):
        if len(input.shape) < 3:
            input = input[np.newaxis, :, :]
        return np.argmax(self.model.predict(input))

    def evaluate_sensor_model(self, model, args):
        if self.dataset is None:
            self._load_dataset()
        if model is None:
            model = self.get_model(input_shape=(self.dataset['x_train'].shape[1], self.dataset['x_train'].shape[2]),
                                   output_shape=self.dataset['y_train'].shape[1], layer_size=args.lstm_layer_size,
                                   dropout=args.dropout)

        model.load_weights(args.lstm_model_weights)
        print(model.evaluate(self.dataset['x_test'], self.dataset['y_test']))

    def _load_dataset(self):
        self.dataset = dict()
        multimodal_dataset = MultimodalDataset()

        train_sns_x, train_sns_y = multimodal_dataset.load_all_sensor_files(
            selected_sensors=['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'],
            sensor_root=args.train_dir)

        train_onehot_y = np.eye(20)[np.squeeze(train_sns_y).astype(int)]

        train_sns_x, train_onehot_y = multimodal_dataset.split_windows(args.sensor_chunk_size, args.sensor_step_size,
                                                                       train_sns_x, train_onehot_y)

        print(train_sns_x.shape)

        test_sns_x, test_sns_y = multimodal_dataset.load_all_sensor_files(
            selected_sensors=['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'],
            sensor_root=args.val_dir)

        test_onehot_y = np.eye(20)[np.squeeze(test_sns_y).astype(int)]

        test_sns_x, test_onehot_y = multimodal_dataset.split_windows(args.sensor_chunk_size, args.sensor_step_size,
                                                                     test_sns_x, test_onehot_y)

        print(test_sns_x.shape)

        self.dataset['x_train'] = train_sns_x
        self.dataset['y_train'] = train_onehot_y
        self.dataset['x_test'] = test_sns_x
        self.dataset['y_test'] = test_onehot_y

    def train_sensor_model(self, args):
        if self.dataset is None:
            self._load_dataset()

        model = self.get_model(input_shape=(self.dataset['x_train'].shape[1], self.dataset['x_train'].shape[2]),
                               output_shape=self.dataset['y_train'].shape[1], layer_size=args.lstm_layer_size,
                               dropout=args.dropout)

        earlystopping = EarlyStopping(patience=20)

        model_checkpoint = ModelCheckpoint('checkpoints/sensor_model.hdf5', save_best_only=True, monitor='val_acc')

        model.fit(self.dataset['x_train'], self.dataset['y_train'], batch_size=args.batch_size, epochs=args.nb_epochs,
                  validation_data=(self.dataset['x_test'], self.dataset['y_test']),
                  callbacks=[earlystopping, model_checkpoint], verbose=2)

    def get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(layer_size, input_shape=input_shape))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='multimodal_dataset/sensor/train')
    a.add_argument("--val_dir", default='multimodal_dataset/sensor/test')
    a.add_argument("--batch_size", default=1000, type=int)
    a.add_argument("--nb_epochs", default=1000, type=int)
    a.add_argument("--dropout", default=0.6, type=int)
    a.add_argument("--sensor_chunk_size", int=15, type=int)
    a.add_argument("--sensor_step_size", int=1, type=int)
    a.add_argument("--lstm_layer_size", int=16, type=int)

    args = a.parse_args()

    sns_agent = SensorAgent()
    sns_agent.train_sensor_model(args)
