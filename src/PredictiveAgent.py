from keras.models import Sequential

class PredictiveAgent(object):

    def predict(self, action_num):
        raise NotImplementedError("Should implement the predict method")

    def _get_model(self, input_shape, output_shape, layer_size=None, optimizer=None, dropout=None):
        raise NotImplementedError("The _get_model function needs to be implemented")

    def _fit_transform(self, model, dataset, epochs=None, batch_size=None, callbacks=[], verbose=1):
        raise NotImplementedError("The _fit_transform function needs to be implemented")

