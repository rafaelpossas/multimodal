import argparse
from keras.callbacks import *
import keras
from keras.optimizers import *
from MultimodalDataset import MultimodalDataset
import math
import datetime as dt

class SensorAgent(object):
    model = None

    def __init__(self, weights=None, network_layer_size=128,
                 sns_chunk_size=10, num_sensors=2, num_classes=20):
        if weights is not None:
            self.model = self.get_model(input_shape=(sns_chunk_size, num_sensors * 3),
                                        output_shape=num_classes, layer_size=network_layer_size)
            self.model.load_weights(weights)

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
        nb_train_samples = MultimodalDataset.get_total_size(args.train_dir)
        nb_val_samples = MultimodalDataset.get_total_size(args.val_dir)
        nb_epoch = args.nb_epochs
        batch_size = int(args.batch_size)
        group_size = args.sensor_chunk_size
        model = self.get_model(input_shape=(args.sensor_chunk_size, args.num_sensors * 3),
                               output_shape=20, layer_size=args.layer_size,
                               dropout=args.dropout,
                               optimizer=args.optimizer)
        save_dir = os.path.join('models', 'sensor', str(args.layer_size), str(args.optimizer))

        os.makedirs(save_dir, exist_ok=True)

        model_checkpoint = ModelCheckpoint(os.path.join(save_dir, 'sns_{val_acc:2f}-{acc:.2f}.hdf5'),
                                           save_best_only=True, monitor='val_acc')

        reduce_lr_on_plateau = ReduceLROnPlateau(min_lr=0.000001, verbose=1, factor=0.5)

        model.fit_generator(
            MultimodalDataset.flow_from_dir(root=args.train_dir, max_frames_per_video=150,
                                            group_size=group_size,
                                            batch_size=args.batch_size, type="sns"),
            steps_per_epoch=math.ceil(nb_train_samples / (batch_size * group_size)),
            epochs=nb_epoch,
            validation_data=MultimodalDataset.flow_from_dir(root=args.val_dir, max_frames_per_video=150,
                                                            group_size=group_size,
                                                            batch_size=args.batch_size, type="sns"),
            validation_steps=math.ceil(nb_val_samples / (batch_size * group_size)),
            callbacks=[model_checkpoint, reduce_lr_on_plateau],
            verbose=2)

    def get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(layer_size, input_shape=input_shape))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='multimodal_dataset/video/splits/train')
    a.add_argument("--val_dir", default='multimodal_dataset/video/splits/test')
    a.add_argument("--batch_size", default=500, type=int)
    a.add_argument("--nb_epochs", default=150, type=int)
    a.add_argument("--dropout", default=0.3, type=int)
    a.add_argument("--sensor_chunk_size", default=10, type=int)
    a.add_argument("--sensor_step_size", default=1, type=int)
    a.add_argument("--layer_size", default=64, type=int)
    a.add_argument("--grid_search", action="store_true")
    a.add_argument("--num_sensors", default=2, type=int)

    args = a.parse_args()

    sns_agent = SensorAgent()
    if args.grid_search:
        optimizers = ['adam', 'adadelta', 'adagrad', 'rmsprop']
        layer_size = ['128', '64', '32', '16']
        epochs = ['150', '200', '250', '300']
        for opt in optimizers:
            for ix, ly_size in enumerate(layer_size):
                print("Training for {}-{}".format(opt, ly_size))
                args.optimizer = opt
                args.layer_size = int(ly_size)
                args.nb_epochs = int(epochs[ix])
                sns_agent.train_sensor_model(args)
    else:
        sns_agent.train_sensor_model(args)
