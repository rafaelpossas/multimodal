
from src.CNNFBF import CNNFBF
from src.LSTM import SensorLSTM
import h5py
from scipy.stats import mode
import numpy as np

from src.MultimodalDataset import MultimodalDataset
from src.PredictiveAgent import PredictiveAgent
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import h5py
import numpy as np
import tensorflow as tf
import keras


class VisionAgent(object):

    model = None

    def __init__(self, model_weights=None, num_classes=20):
        self.num_classes = num_classes
        if model_weights is not None:
            self.model = self._get_model(model_weights)
            self.model.load_weights(model_weights)

    def predict(self, x, num_samples=15):
        if self.model is not None:
            pred = self.model.predict(x)
            pred = np.argmax(pred, axis=1)
            pred = pred.reshape((int(pred.shape[0] / num_samples), num_samples, 1))
            return [mode(arr.flatten())[0][0] for arr in pred][0]
        else:
            raise Exception("The CNN model needs to be provided")

    def _get_model(self, weights=None):
        base_model = InceptionV3(include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(self.num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        if weights is not None:
            model.load_weights(weights)

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _fit_transform(self, model, dataset, epochs=100, batch_size=2048, callbacks=[], verbose=1):
        model.fit(dataset.x_train, dataset.y_train, validation_split=0.1,
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        scores = model.evaluate(dataset.x_test, dataset.y_test, verbose=verbose)
        return scores


#
# def get_image_prediction(x, cnn_model=None, num_samples=15):
#     if cnn_model is not None:
#         pred = cnn_model.predict(x)
#         pred = np.argmax(pred, axis=1)
#         pred = pred.reshape((int(pred.shape[0]/num_samples), num_samples, 1))
#         return np.asarray([mode(arr.flatten())[0][0] for arr in pred])
#     else:
#         raise Exception("The CNN model needs to be provided")
#
#
# def get_sensor_prediction(x, lstm_model=None, num_samples=5):
#     if lstm_model is not None:
#         pred = lstm_model.predict(x)
#         pred = np.argmax(pred, axis=1)
#         return pred
#     else:
#         raise Exception("The CNN model needs to be provided")
#
# if __name__ == '__main__':
#     vision = VisionAgent()
#     lstm = SensorLSTM()
#     num_classes = 20
#     weights_file = None
#     f_test = h5py.File('multimodal_test.hdf5')
#
#     x_img = f_test['x_img']
#     y_img = f_test['y_img']
#
#     x_sns = f_test['x_sns']
#     y_sns = f_test['y_sns']
#
#     print(x_img.shape)
#     print(y_img.shape)
#
#     print(x_sns.shape)
#     print(y_sns.shape)
#
#     model_cnn = vision._get_model(num_classes=20, weights='checkpoints/inception.029-1.08.hdf5')
#     #x, y = next(cnn.image_generator(f_test, batch_size=2))
#     #print(model_cnn.evaluate(x, y))
#
#     #pred_cnn = get_image_prediction(x, model_cnn)
#     pred_cnn = model_cnn.predict_generator(generator=vision.image_generator(f_test, 24), steps=50,verbose=1)
#     print(pred_cnn)
#
#     model_lstm = lstm.get_model(input_shape=(x_sns.shape[1], x_sns.shape[2]),
#                                 output_shape=num_classes, dropout=0.4, layer_size=128,
#                                 optimizer='rmsprop')
#
#     model_lstm.load_weights('sensor_model.hdf5')
#     x = x_sns[:]
#     pred_lstm = get_sensor_prediction(x, lstm_model=model_lstm)
#     print(pred_lstm)
#
#     #print(y_img[:10])
#     #print(y_sns[:10])
#
#     with h5py.File('predictions.hdf5', "w") as hf:
#         hf.create_dataset("pred_img", data=pred_cnn)
#         hf.create_dataset("pred_sns", data=pred_lstm)
