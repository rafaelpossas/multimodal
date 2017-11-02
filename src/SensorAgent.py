import argparse
from keras.callbacks import *
import keras
from keras.optimizers import *
from MultimodalDataset import MultimodalDataset
from VuzixDataset import VuzixDataset
import math
from tensorflow.python.platform import gfile
import datetime as dt
from keras.models import load_model


class SensorAgent(object):
    model = None

    def __init__(self, weights=None, network_layer_size=128,
                 sns_chunk_size=10, num_sensors=2, num_classes=20,
                 tf_input=None, tf_output=None):
        if weights is not None:
            self.model = self.get_model(input_shape=(sns_chunk_size, num_sensors * 3),
                                        output_shape=num_classes, layer_size=network_layer_size)
            self.model.load_weights(weights)
            self.input_tf, self.output_tf = tf_input, tf_output

    def predict(self, input=None):
        if len(input.shape) < 3:
            input = input[np.newaxis, :, :]
        outputs = self.model.predict(input)
        return np.argmax(outputs)

    def predict_from_tf(self, input, session=None):
        if len(input.shape) < 3:
            input = input[np.newaxis, :, :]

        return np.argmax(session.run([self.output_tf], {self.input_tf: input}))

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

        if args.dataset == 'vuzix':
            flow_from_dir = VuzixDataset.flow_from_dir
            max_frames_per_video = 4500
        else:
            flow_from_dir = MultimodalDataset.flow_from_dir
            max_frames_per_video = 150

        nb_epoch = args.nb_epochs
        batch_size = int(args.batch_size)
        group_size = args.sensor_chunk_size

        nb_val_samples = MultimodalDataset.get_total_size(args.val_dir, max_frames_per_video)

        nb_train_samples = MultimodalDataset.get_total_size(args.train_dir, max_frames_per_video)

        model = self.get_model(input_shape=(args.sensor_chunk_size, args.num_sensors * 3),
                               output_shape=20, layer_size=args.layer_size,
                               dropout=args.dropout,
                               optimizer=args.optimizer)

        save_dir = os.path.join('models', 'sensor', str(args.layer_size), str(args.optimizer))

        os.makedirs(save_dir, exist_ok=True)

        model_checkpoint = ModelCheckpoint(os.path.join(save_dir, args.dataset + '_sns_{val_acc:2f}-{acc:.2f}.hdf5'),
                                           save_best_only=True, monitor='val_acc')

        reduce_lr_on_plateau = ReduceLROnPlateau(min_lr=0.000001, verbose=1, patience=50, factor=0.1)

        model.fit_generator(
            flow_from_dir(root=args.train_dir, max_frames_per_video=max_frames_per_video,
                          group_size=group_size,
                          batch_size=batch_size, type="sns"),
            steps_per_epoch=math.floor(nb_train_samples/(args.sensor_chunk_size * batch_size)),
            epochs=nb_epoch,
            validation_data=flow_from_dir(root=args.val_dir, max_frames_per_video=max_frames_per_video,
                                          group_size=group_size,
                                          batch_size=batch_size, type="sns"),
            validation_steps=math.floor(nb_val_samples/(args.sensor_chunk_size * batch_size)),
            callbacks=[model_checkpoint, reduce_lr_on_plateau],
            verbose=2)

    def evaluate(self, args):

        if args.dataset == 'vuzix':
            max_frames_per_video = 4500
            flow_from_dir = VuzixDataset.flow_from_dir
        else:
            max_frames_per_video = 150
            flow_from_dir = MultimodalDataset.flow_from_dir

        model = load_model(args.weights)

        nb_val_samples = MultimodalDataset.get_total_size(args.val_dir, max_frames_per_video)

        nb_train_samples = MultimodalDataset.get_total_size(args.train_dir, max_frames_per_video)

        print("Evaluating on Test Set")
        result = model.evaluate_generator(flow_from_dir(root=args.val_dir, group_size=args.sensor_chunk_size,
                                                        batch_size=nb_val_samples/args.sensor_chunk_size,
                                                        max_frames_per_video=max_frames_per_video, type="sns",
                                                        shuffle_arrays=False),
                                          steps=1)
        print(result)

        print("Evaluating on Train Set")
        result = model.evaluate_generator(flow_from_dir(root=args.train_dir, group_size=args.sensor_chunk_size,
                                                        batch_size=nb_train_samples/args.sensor_chunk_size,
                                                        max_frames_per_video=max_frames_per_video,
                                                        type="sns", shuffle_arrays=False),
                                          steps=1)
        print(result)

    def get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(layer_size, input_shape=input_shape))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='multimodal_dataset/video/splits/train')
    a.add_argument("--val_dir", default='multimodal_dataset/video/splits/test')
    a.add_argument("--batch_size", default=100, type=int)
    a.add_argument("--nb_epochs", default=500, type=int)
    a.add_argument("--dropout", default=0.6, type=float)
    a.add_argument("--sensor_chunk_size", default=10, type=int)
    a.add_argument("--sensor_step_size", default=1, type=int)
    a.add_argument("--layer_size", default=128, type=int)
    a.add_argument("--grid_search", action="store_true")
    a.add_argument("--evaluate", action="store_true")
    a.add_argument("--train", action="store_true")
    a.add_argument("--num_sensors", default=2, type=int)
    a.add_argument("--dataset", default='multimodal', type=str)
    a.add_argument("--optimizer", default='rmsprop', type=str)
    a.add_argument("--weights", default="sensor_model_full.hdf5")

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

    if args.train:
        sns_agent.train_sensor_model(args)

    if args.evaluate:
        sns_agent.evaluate(args)
