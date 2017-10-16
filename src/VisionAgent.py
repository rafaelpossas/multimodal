
from scipy.stats import mode

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.models import Model
from MultimodalDataset import MultimodalDataset
import matplotlib.pyplot as plt
import numpy as np
import h5py
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import argparse
import sys
import os
import math
import glob
import cv2

class VisionAgent(object):

    model = None
    FC_SIZE = 32
    NB_IV3_LAYERS_TO_FREEZE = 172
    IM_WIDTH = 224
    IM_HEIGHT = 224

    def __init__(self, model_weights=None, num_classes=20, train_root='multimodal_dataset/video/images/train',
                 test_root='multimodal_dataset/video/images/test'):
        self.num_classes = num_classes
        self.train_root = train_root
        self.test_root = test_root
        if model_weights is not None:
            self.model = self._get_model(model_weights)
            self.model.load_weights(model_weights)

    def setup_to_transfer_learn(self,model, base_model):
        """Freeze all layers and compile the model"""
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def add_new_last_layer(self,base_model, nb_classes):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        x = Dense(self.FC_SIZE, activation='relu')(x)  # new FC layer, random init
        x = Dropout(0.5)(x)
        predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def setup_to_finetune(self, model, fine_tune_lr=0.0001):
        """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
        note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
        Args:
          model: keras model
        """
        for layer in model.layers[:self.NB_IV3_LAYERS_TO_FREEZE]:
            layer.trainable = False
        for layer in model.layers[self.NB_IV3_LAYERS_TO_FREEZE:]:
            layer.trainable = True
        model.compile(optimizer=SGD(lr=fine_tune_lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, x, num_samples=15):
        if self.model is not None:
            pred = self.model.predict(x)
            pred = np.argmax(pred, axis=1)
            pred = pred.reshape((int(pred.shape[0] / num_samples), num_samples, 1))
            return [mode(arr.flatten())[0][0] for arr in pred][0]
        else:
            raise Exception("The CNN model needs to be provided")

    def flow_from_dir(self, root, num_frames=10, max_frames_per_seq=450):
        act_str_arr = MultimodalDataset.dataset.get_activities_by_index(range(1, 21))
        x = []
        y = []
        while True:
            for act_str in act_str_arr:
                path = os.path.join(root, act_str)
                all_seq = glob.glob(os.path.join(path, '*'))
                for seq_ix, seq in enumerate(sorted(all_seq)):
                    files = glob.glob(os.path.join(seq, '*.jpg'))
                    for img_ix, img in enumerate(sorted(files)):
                        if img_ix < max_frames_per_seq:
                            file_name = img.split(os.path.sep)[-1]
                            dir_downsampled = os.path.join(seq, 'downsampled')
                            full_path_downsampled = os.path.join(dir_downsampled, file_name)

                            cur_img = cv2.imread(full_path_downsampled)

                            cur_img = cur_img / 255.0

                            x.append(cur_img)

                            y.append(self.dataset.activity_dict[act_str][0])

                            if len(x) == num_frames:
                                yield np.array(x), np.eye(20)[np.array(y).astype(int)]
                                x, y = ([], [])

    def get_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        model = self.add_new_last_layer(base_model, self.num_classes)
        return base_model, model

    def train(self,args):
        """Use transfer learning and fine-tuning to train a network on a new dataset"""
        nb_train_samples = MultimodalDataset.get_total_size(self.train_root)
        nb_val_samples = MultimodalDataset.get_total_size(self.test_root)
        nb_epoch_fine_tune = int(args.nb_epoch_fine_tune)
        nb_epoch_train = int(args.nb_epoch_train)
        batch_size = int(args.batch_size)

        # data prep
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            # rotation_range=30,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
            args.train_dir,
            target_size=(self.IM_WIDTH, self.IM_HEIGHT),
            batch_size=batch_size,
            save_format='jpg'
        )

        validation_generator = test_datagen.flow_from_directory(
            args.val_dir,
            target_size=(self.IM_WIDTH, self.IM_HEIGHT),
            batch_size=batch_size,
            save_format='jpg'
        )

        # setup model
        base_model, model = self.get_model()

        # fine-tuning
        self.setup_to_finetune(model,args.fine_tune_lr)

        history_ft = model.fit_generator(
            train_generator,
            steps_per_epoch=math.ceil(nb_train_samples/batch_size),
            epochs=nb_epoch_fine_tune,
            validation_data=validation_generator,
            validation_steps=math.ceil(nb_val_samples/batch_size),
            class_weight='auto')

        # transfer learning
        self.setup_to_transfer_learn(model, base_model)

        history_tl = model.fit_generator(
            train_generator,
            epochs=nb_epoch_train,
            steps_per_epoch=math.ceil(nb_train_samples/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(nb_val_samples/batch_size),
            class_weight='auto')



        model.save(args.output_model_file)

        if args.plot:
            self.plot_training(history_ft)

    def plot_training(self, history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training and validation accuracy')

        plt.figure()
        plt.plot(epochs, loss, 'r.')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training and validation loss')
        plt.show()


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='multimodal_dataset/video/images/train')
    a.add_argument("--val_dir", default='multimodal_dataset/video/images/test')
    a.add_argument("--nb_epoch_fine_tune", default=3)
    a.add_argument("--nb_epoch_train", default=10)
    a.add_argument('--fine_tune_lr', default=0.0001)
    a.add_argument("--batch_size", default=150)
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    a.add_argument("--plot", action="store_true")

    vision_agent = VisionAgent()
    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    vision_agent.train(args)
#
# if __name__ == '__main__':
#     vision_agent = VisionAgent()
#     vs_model = vision_agent._get_model()
#     train_file = h5py.File('data/multimodal_full_train.hdf5')
#     test_file = h5py.File("data/multimodal_full_test.hdf5")
#     vs_model.fit_generator(
#         vision_agent.image_generator(train_file, batch_size=1, num_frames=25),
#         steps_per_epoch=1,
#         max_queue_size=1,
#         epochs=100)

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
