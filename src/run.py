
import h5py
import numpy as np
from arguments import args
from MultimodalDataset import MultimodalDataset
from SensorAgent import SensorAgent
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau


dataset = dict()


def load_sensor_dataset():

    multimodal_dataset = MultimodalDataset()

    train_sns_x, train_sns_y = multimodal_dataset.load_all_sensor_files(selected_sensors=['accx','accy','accz','gyrx','gyry','gyrz'],
                                                                        sensor_root='multimodal_dataset/sensor/train')

    train_onehot_y = np.eye(20)[np.squeeze(train_sns_y).astype(int)]

    train_sns_x, train_onehot_y = multimodal_dataset.split_windows(args.sensor_chunk_size, args.sensor_step_size, train_sns_x, train_onehot_y)

    print(train_sns_x.shape)

    test_sns_x, test_sns_y =  multimodal_dataset.load_all_sensor_files(selected_sensors=['accx','accy','accz','gyrx','gyry','gyrz'],
                                                                        sensor_root='multimodal_dataset/sensor/test')
    test_onehot_y = np.eye(20)[np.squeeze(test_sns_y).astype(int)]

    test_sns_x, test_onehot_y = multimodal_dataset.split_windows(args.sensor_chunk_size, args.sensor_step_size, test_sns_x, test_onehot_y)

    print(test_sns_x.shape)

    dataset['x_train'] = train_sns_x
    dataset['y_train'] = train_onehot_y
    dataset['x_test'] = test_sns_x
    dataset['y_test'] = test_onehot_y


def train_sensor_model():
    load_sensor_dataset()
    sensor_agent = SensorAgent(input_shape=(dataset['x_train'].shape[1], dataset['x_train'].shape[2]),
                               output_shape=dataset['y_train'].shape[1],
                               layer_size=args.lstm_layer_size)

    earlystopping = EarlyStopping(patience=20)

    model_checkpoint = ModelCheckpoint('checkpoints/sensor_model.hdf5', save_best_only=True, monitor='val_acc')

    sensor_agent.model.fit(dataset['x_train'], dataset['y_train'], batch_size=1000, epochs=1000,
                           validation_data=(dataset['x_test'], dataset['y_test']),
                           callbacks=[earlystopping, model_checkpoint], verbose=2)


def evaluate_sensor_model():
    if dataset is None:
        load_sensor_dataset()

    sensor_agent = SensorAgent(input_shape=(dataset['x_train'].shape[1], dataset['x_train'].shape[2]),
                               output_shape=dataset['y_train'].shape[1],
                               layer_size=args.lstm_layer_size)

    sensor_agent.model.load_weights(args.lstm_model_weights)
    print(sensor_agent.model.evaluate(dataset['x_test'], dataset['y_test']))


if __name__ == "__main__":
    train_sensor_model()