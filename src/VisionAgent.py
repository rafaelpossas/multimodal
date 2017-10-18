
from scipy.stats import mode
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.models import Model
from MultimodalDataset import MultimodalDataset
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import argparse
import sys
import os
import math
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


class VisionAgent(object):

    model = None
    NB_LAYERS_TO_FREEZE = None
    NB_IV3_LAYERS_TO_FREEZE = 54
    IM_WIDTH = 224
    IM_HEIGHT = 224
    activity_dict = {
        'act01': (0, 'walking'), 'act02': (1, 'walking upstairs'), 'act03': (2, 'walking downstairs'),
        'act04': (3, 'riding elevator up'), 'act05': (4, 'riding elevator down'), 'act06': (5, 'riding escalator up'),
        'act07': (6, 'riding escalator down'), 'act08': (7, 'sitting'), 'act09': (8, 'eating'),
        'act10': (9, 'drinking'),
        'act11': (10, 'texting'), 'act12': (11, 'phone calls'), 'act13': (12, 'working on pc'),
        'act14': (13, 'reading'),
        'act15': (14, 'writing sentences'), 'act16': (15, 'organizing files'), 'act17': (16, 'running'),
        'act18': (17, 'push-ups'),
        'act19': (18, 'sit-ups'), 'act20': (19, 'cycling')
    }

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

    def add_new_last_layer(self,base_model, nb_classes, fc_size=64, dropout=0.2):
        """Add last layer to the convnet
        Args:
          base_model: keras model excluding top
          nb_classes: # of classes
        Returns:
          new keras model with last layer
        """
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(fc_size, activation='relu')(x)  # new FC layer, random init
        x = Dropout(dropout)(x)
        predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def setup_to_finetune(self, model, fine_tune_lr=0.0001):
        """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
        note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
        Args:
          model: keras model
        """
        for layer in model.layers[:self.NB_LAYERS_TO_FREEZE]:
            layer.trainable = False
        for layer in model.layers[self.NB_LAYERS_TO_FREEZE:]:
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
        all_directories = glob.glob(os.path.join(root, '*', '*'))
        np.random.shuffle(all_directories)
        x = []
        y = []
        while True:
            for dir in all_directories:
                activity = dir.split(os.path.sep)[-2]
                files = glob.glob(os.path.join(dir, '*.jpg'))
                for img_ix, img in enumerate(sorted(files)):
                    if img_ix < max_frames_per_seq:

                        cur_img = cv2.resize(cv2.imread(img), (224, 224)).astype('float')

                        cur_img /= 255.
                        cur_img -= 0.5
                        cur_img *= 2.

                        x.append(cur_img)

                        y.append(self.activity_dict[activity][0])

                        if len(x) == num_frames:
                            #print(img)
                            yield np.array(x), np.eye(20)[np.array(y).astype(int)]
                            x, y = ([], [])

    def get_trainable_layers(self, model):
        count = 0
        for layer in model.layers:
            weights = layer.weights
            if weights:
                count+=1
        return count

    def get_model(self, fc_size=64, dropout=0.5, architecture="mobilenet"):
        #base_model = InceptionV3(weights='imagenet', include_top=False)  # include_top=False excludes final FC layer
        if architecture == "mobilenet":
            base_model = MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False)
            self.NB_LAYERS_TO_FREEZE = 54
        if architecture == "resnet":
            base_model = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
            self.NB_LAYERS_TO_FREEZE = 163
        if architecture == "inception":
            base_model = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
            self.NB_LAYERS_TO_FREEZE = 249

        model = self.add_new_last_layer(base_model, self.num_classes, fc_size, dropout)
        return base_model, model

    def train(self,args):
        """Use transfer learning and fine-tuning to train a network on a new dataset"""
        nb_train_samples = MultimodalDataset.get_total_size(self.train_root)
        nb_val_samples = MultimodalDataset.get_total_size(self.test_root)
        nb_epoch_fine_tune = int(args.nb_epoch_fine_tune)
        nb_epoch_transferlearn = int(args.nb_epoch_transferlearn)
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
        base_model, model = self.get_model(fc_size=args.fc_size, dropout=args.dropout, architecture=args.architecture)
        # fine-tuning

        self.setup_to_transfer_learn(model, base_model)
        if not args.load_fine_tuned_model and args.pre_trained_model is None:

            history_ft = model.fit_generator(
                self.flow_from_dir(args.train_dir,args.batch_size),
                steps_per_epoch=math.ceil(nb_train_samples/batch_size),
                epochs=nb_epoch_transferlearn,
                validation_data=self.flow_from_dir(args.val_dir, args.batch_size),
                validation_steps=math.ceil(nb_val_samples/batch_size),
                class_weight='auto')

            model.save_weights('models/vision/'+args.architecture+'_finetune.hdf5')
            # transfer learning

        else:
            if args.pre_trained_model is None:
                model.load_weights('models/vision/'+args.architecture+'_finetune.hdf5')
            else:
                model.load_weights(args.pre_trained_model)

        self.setup_to_finetune(model, args.fine_tune_lr)

        checkpointer = ModelCheckpoint(
            filepath='models/vision/'+args.architecture+'.{epoch:03d}-{acc:2f}-{val_acc:.2f}.hdf5',
            verbose=0,
            monitor='val_acc',
            save_best_only=True)

        history_tl = model.fit_generator(
            train_generator,
            epochs=nb_epoch_fine_tune,
            steps_per_epoch=math.ceil(nb_train_samples/batch_size),
            validation_data=validation_generator,
            validation_steps=math.ceil(nb_val_samples/batch_size),
            class_weight='auto',
            callbacks=[checkpointer])

        if args.plot:
            self.plot_training(history_tl)

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
    a.add_argument("--nb_epoch_fine_tune", default=20)
    a.add_argument("--nb_epoch_transferlearn", default=10)
    a.add_argument('--fine_tune_lr', default=0.0001)
    a.add_argument("--load_fine_tuned_model", default=None)
    a.add_argument("--pre_trained_model", default=None)
    a.add_argument("--batch_size", default=150)
    a.add_argument("--plot", action="store_true")
    a.add_argument("--dropout", default=0.3)
    a.add_argument("--fc_size", default=1024)
    a.add_argument("--architecture", default="resnet")
    vision_agent = VisionAgent()
    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    vision_agent.train(args)