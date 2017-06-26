import random
import numpy as np
import h5py
from src.LSTM import SensorLSTM
from src.MultimodalDataset import MultimodalDataset
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop

num_frames_per_sample = 15
def image_generator(file, batch_size):
    while True:
        total_size = file['x'].shape[0]
        index = random.sample(range(0, total_size), batch_size)
        x = file['x_img'][sorted(index)]
        y = file['y_img'][sorted(index)]
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        y = np.eye(20)[np.repeat(y, num_frames_per_sample)]
        x = x.astype("float") / 255.0
        yield x, y

def sensor_generator(file, batch_size):
    while True:
        total_size = file['x_sns'].shape[0]
        index = random.sample(range(0, total_size), batch_size)
        x = file['x_sns'][sorted(index)]
        y = file['y_sns'][sorted(index)]
        return x, y


def train_model():
    chunk_size = 5
    step_size = 1

    #scaler = MinMaxScaler()
    train_file = h5py.File('multimodal_train.hdf5')
    test_file = h5py.File('multimodal_test.hdf5')
    dataset = MultimodalDataset()

    train_sns_x = train_file['x_sns'][:]
    train_sns_y = train_file['y_sns'][:]
    train_onehot_y = np.eye(20)[train_sns_y]

    train_original_x = train_sns_x.reshape(
        (int(train_sns_x.shape[0] / 30), train_sns_x.shape[1] * 30, train_sns_x.shape[2]))
    train_original_one_hot_y = np.array([y for ix, y in enumerate(train_onehot_y) if ix % 30 == 0])

    train_sns_x, train_onehot_y = dataset.split_windows(chunk_size, step_size, train_original_x, train_original_one_hot_y)

    train_original_shape = train_sns_x.shape
    train_reshaped_shape = (train_sns_x.shape[0]*train_sns_x.shape[1], train_sns_x.shape[2])
    #train_sns_x = scaler.fit_transform(train_sns_x.reshape(train_reshaped_shape)).reshape(train_original_shape)
    print(train_sns_x.shape)


    test_sns_x = test_file['x_sns'][:]
    test_sns_y = test_file['y_sns'][:]
    test_onehot_y = np.eye(20)[test_sns_y]

    test_original_x = test_sns_x.reshape(
        (int(test_sns_x.shape[0] / 30), test_sns_x.shape[1] * 30, test_sns_x.shape[2]))
    test_original_one_hot_y = np.array([y for ix, y in enumerate(test_onehot_y) if ix % 30 == 0])

    test_sns_x, test_onehot_y = dataset.split_windows(chunk_size, step_size, test_original_x, test_original_one_hot_y)

    test_original_shape = test_sns_x.shape
    test_reshaped_shape = (test_sns_x.shape[0]*test_sns_x.shape[1], test_sns_x.shape[2])
    #test_sns_x = scaler.transform(test_sns_x.reshape(test_reshaped_shape)).reshape(test_original_shape)
    print(test_sns_x.shape)

    sensor_lstm = SensorLSTM()
    earlystopping = EarlyStopping(patience=10)
    #reduce_lr = ReduceLROnPlateau(patience=20, min_lr=0.000001, verbose=1)
    rmsprop = RMSprop()
    model_lstm = sensor_lstm.get_model(input_shape=(train_sns_x.shape[1], train_sns_x.shape[2]),
                                       output_shape=train_onehot_y.shape[1], dropout=0.4, layer_size=128, optimizer=rmsprop)
    #model_lstm.load_weights('models/65.00_accx_accy_accz_rmsprop_64_75_0.4.hdf5')
    model_lstm.fit(train_sns_x, train_onehot_y, batch_size=48, epochs=1000,
                   validation_data=(test_sns_x, test_onehot_y), callbacks=[earlystopping], verbose=2)
    print(model_lstm.evaluate(test_sns_x, test_onehot_y))
    model_lstm.save_weights('sensor_model.hdf5')

def create_dataset(sensors):
    dataset = MultimodalDataset()
    dataset.load_multimodal_dataset(450, 450, '../multimodal_dataset/video/images/train',
                                    sensor_root='../multimodal_dataset/sensor/',
                                    sensors=sensors,
                                     output_file='multimodal_full_train.hdf5')

    dataset.load_multimodal_dataset(450, 450, '../multimodal_dataset/video/images/test',
                                    sensor_root='../multimodal_dataset/sensor/',
                                    sensors=sensors,
                                     output_file='multimodal_full_test.hdf5')
if __name__=='__main__':
    create_dataset(['grax','gray','graz','gyrx','gyry','gyrz','lacx','lacy','lacz'])
    #train_model()
