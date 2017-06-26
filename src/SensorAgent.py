from src.MultimodalDataset import MultimodalDataset
from src.PredictiveAgent import PredictiveAgent
import h5py
import numpy as np
import tensorflow as tf
import keras


class SensorAgent(PredictiveAgent):

    model = None

    def __init__(self, input_shape=None, output_shape=None, model_weights=None, layer_size=128):

        if model_weights is not None:
            self.model = self._get_model(input_shape=input_shape, output_shape=output_shape, layer_size=layer_size)
            self.model.load_weights(model_weights)
            self.intermediate_model = self._get_intermediate_model()

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

    def _get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(layer_size, input_shape=input_shape))
        model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def _fit_transform(self, model, dataset, epochs=100, batch_size=2048, callbacks=[], verbose=1):
        model.fit(dataset.x_train, dataset.y_train, validation_split=0.1,
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        scores = model.evaluate(dataset.x_test, dataset.y_test, verbose=verbose)
        return scores


