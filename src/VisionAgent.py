
from src.CNNFBF import CNNFBF
from src.LSTM import SensorLSTM
import h5py
from scipy.stats import mode
import numpy as np


def get_image_prediction(x, cnn_model=None, num_samples=15):
    if cnn_model is not None:
        pred = cnn_model.predict(x)
        pred = np.argmax(pred, axis=1)
        pred = pred.reshape((int(pred.shape[0]/num_samples), num_samples, 1))
        return np.asarray([mode(arr.flatten())[0][0] for arr in pred])
    else:
        raise Exception("The CNN model needs to be provided")


def get_sensor_prediction(x, lstm_model=None, num_samples=5):
    if lstm_model is not None:
        pred = lstm_model.predict(x)
        pred = np.argmax(pred, axis=1)
        return pred
    else:
        raise Exception("The CNN model needs to be provided")

if __name__ == '__main__':
    cnn = CNNFBF()
    lstm = SensorLSTM()
    num_classes=20
    weights_file = None
    f_test = h5py.File('multimodal_test.hdf5')

    x_img = f_test['x_img']
    y_img = f_test['y_img']

    x_sns = f_test['x_sns']
    y_sns = f_test['y_sns']

    print(x_img.shape)
    print(y_img.shape)

    print(x_sns.shape)
    print(y_sns.shape)

    model_cnn = cnn.get_model(num_classes=20, weights='checkpoints/inception.029-1.08.hdf5')
    #x, y = next(cnn.image_generator(f_test, batch_size=2))
    x = x_img[:]
    y = y_img[:]
    num_frames_per_sample = x_img.shape[1]
    x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
    x = x.astype("float") / 255.0
    y = np.eye(20)[np.repeat(y, num_frames_per_sample)]
    #print(model_cnn.evaluate(x, y))

    pred_cnn = get_image_prediction(x, model_cnn)
    print(pred_cnn)

    model_lstm = lstm.get_model(input_shape=(x_sns.shape[1], x_sns.shape[2]),
                                output_shape=num_classes, dropout=0.4, layer_size=128, optimizer='rmsprop')
    model_lstm.load_weights('sensor_model.hdf5')
    x = x_sns[:]
    pred_lstm = get_sensor_prediction(x, lstm_model=model_lstm)
    print(pred_lstm)

    #print(y_img[:10])
    #print(y_sns[:10])

    with h5py.File('predictions.hdf5', "w") as hf:
        hf.create_dataset("pred_img", data=pred_cnn)
        hf.create_dataset("pred_sns", data=pred_lstm)
