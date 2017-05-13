from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from src.PredictiveAgent import PredictiveAgent
import numpy as np


class SensorAgent(PredictiveAgent):

    model = None

    def __init__(self, weights_path=None, model=None, timesteps=150, sensors=1, axis=3, num_classes=None):

        if model is None or type(model) is not Sequential:
            print("Standard model not found, creating a new one")
            self.model = self._get_model(input_shape=(timesteps, sensors * axis), output_shape=num_classes,
                                    layer_size=150, optimizer='rmsprop', dropout=0.2)
            self.model.load_weights(weights_path)
        else:
            self.model.load_weights(weights_path)

    def predict(self, input=None):
        return np.argmax(self.model.predict(input))

    def _get_model(self, input_shape, output_shape, layer_size=128, optimizer='rmsprop', dropout=0.2):
        model = Sequential()
        model.add(LSTM(layer_size, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(output_shape, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def _fit_transform(self, model, dataset, epochs=100, batch_size=2048, callbacks=[], verbose=1):
        model.fit(dataset.x_train, dataset.y_train, validation_split=0.1,
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose)
        scores = model.evaluate(dataset.x_test, dataset.y_test, verbose=verbose)
        return scores
