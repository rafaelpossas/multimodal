from MultimodalDataset import MultimodalDataset
import random
import numpy as np
import h5py
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

if __name__=='__main__':
    # file = h5py.File('multimodal_train.hdf5')
    # sensor_generator(file, 10)
    dataset = MultimodalDataset()
    dataset.load_multimodal_dataset(15, 15, '../multimodal_dataset/video/images/train',
                                    sensor_root='../multimodal_dataset/sensor/',
                                    sensors=['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz'],
                                    output_file='multimodal_train.hdf5')