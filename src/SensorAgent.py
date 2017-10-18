import h5py
import numpy as np
import tensorflow as tf
import keras
from MultimodalDataset import MultimodalDataset


class SensorAgent(object):

    model = None

    def __init__(self, input_shape=None, output_shape=None, model_weights=None, layer_size=128):

        self.model = self.get_model(input_shape=input_shape, output_shape=output_shape, layer_size=layer_size)
        self.intermediate_model = self._get_intermediate_model()

        if model_weights is not None:
            self.model.load_weights(model_weights)



    def get_state_for_input(self, input):
        if len(input.shape) < 3:
            input = input[np.newaxis, :, :]
        return self.intermediate_model.predict(input)

    def _get_intermediate_model(self):
        intermediate_model = keras.models.Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        return intermediate_model

    def predict(self, input=None):
        if len(input.shape) < 3:
            input = input[np.newaxis, :, :]
        return np.argmax(self.model.predict(input))

    def get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(layer_size, input_shape=input_shape))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model



# if __name__ == "__main__":
#     multimodal_dataset = MultimodalDataset()
#     dataset_file = h5py.File("data/multimodal_full_test.hdf5")
#     sns_x = dataset_file['x_sns'][:]
#     sns_y = dataset_file['y_sns'][:]
#     onehot_y = np.eye(20)[sns_y]
#     total_size = len(sns_x)
#     sns_x, onehot_y = multimodal_dataset.split_windows(50, 1, sns_x, onehot_y)
#     sensor_agent = SensorAgent(model_weights=None,
#                                input_shape=(5, sns_x.shape[2]),
#                                output_shape=onehot_y.shape[1])