
from scipy.stats import mode

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
import h5py

class VisionAgent(object):

    model = None

    def __init__(self, model_weights=None, num_classes=20):
        self.num_classes = num_classes
        if model_weights is not None:
            self.model = self._get_model(model_weights)
            self.model.load_weights(model_weights)

    def image_generator(self, file, batch_size, num_frames):
        import random
        current_index = 0
        total_size = file['x_img'].shape[0]
        num_frames_per_sample = file['x_img'].shape[1]
        batch_index = 0
        while True:
            if current_index >= total_size:
                current_index = 0

            index = range(current_index, current_index+batch_size)
            x = file['x_img'][index]
            y = file['y'][index]
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
            y = np.eye(20)[np.repeat(y, num_frames_per_sample).astype(int)]
            x = x / 255.0

            rand_ix = [random.randint(0, total_size) for _ in range(num_frames)]
            frames = x[rand_ix]
            frames_y = y[rand_ix]
            batch_index += num_frames
            print("Batch Index {}".format(batch_index))


            current_index += batch_size

            yield frames, frames_y

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

if __name__ == '__main__':
    vision_agent = VisionAgent()
    vs_model = vision_agent._get_model()
    train_file = h5py.File('data/multimodal_full_train.hdf5')
    test_file = h5py.File("data/multimodal_full_test.hdf5")
    vs_model.fit_generator(
        vision_agent.image_generator(train_file, batch_size=1, num_frames=25),
        steps_per_epoch=1,
        max_queue_size=1,
        epochs=100)

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
