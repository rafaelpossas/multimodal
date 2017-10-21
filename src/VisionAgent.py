
from scipy.stats import mode

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.callbacks import *
import keras.backend as K

from MultimodalDataset import MultimodalDataset
from VuzixDataset import VuzixDataset

import argparse
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class VisionAgent(object):

    model = None
    NB_LAYERS_TO_FREEZE = None
    NB_IV3_LAYERS_TO_FREEZE = 54
    IM_WIDTH = 224
    IM_HEIGHT = 224
    NUM_CHANNELS = 3

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

    def __init__(self, model_weights=None, train_root='multimodal_dataset/video/images/train',
                 test_root='multimodal_dataset/video/images/test'):
        self.train_root = train_root
        self.test_root = test_root
        if model_weights is not None:
            #self.model = self.get_model(model_weights)
            self.model.load_weights(model_weights)

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

    def setup_to_transfer_learn(self,model, base_model):
        """Freeze all layers and compile the model"""
        if base_model is not None:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers[:-2]:
                layer.trainable = False
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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

    def intermediate_lrcn_model(self, args):
        _, model = self.get_fbf_model(args)

        if args.fbf_model_weights != "":
            model.load_weights(args.fbf_model_weights)

        return model

    def get_lrcn_model(self, args):
        K.set_learning_phase(0)

        base_model = self.intermediate_lrcn_model(args)
        base_model = Model(inputs=base_model.input, outputs=base_model.layers[-4].output)
        base_model.trainable = False

        x = Input(shape=(None, self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS))
        td_base = TimeDistributed(base_model)(x)
        lstm = LSTM(args.lstm_size, return_sequences=False)(td_base)
        dropout = Dropout(args.dropout)(lstm)
        y = Dense(args.num_classes, activation='softmax')(dropout)

        model = Model(inputs=[x], outputs=y)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return base_model, model

    def train_lrcn(self, args):

        train_root = args.train_dir
        test_root = args.val_dir
        group_size = args.fps
        batch_size = args.batch_size

        total_train_size = MultimodalDataset.get_total_size(train_root)
        total_test_size = MultimodalDataset.get_total_size(test_root)
        steps_per_epoch = math.ceil((total_train_size / (group_size * args.batch_size)))
        steps_per_epoch_val = math.ceil((total_test_size / (group_size * args.batch_size)))

        base_model, model = self.get_lrcn_model(args)

        self.setup_to_transfer_learn(model, base_model)

        model.fit_generator(
            MultimodalDataset.flow_image_from_dir(root=train_root, max_frames_per_video=450, batch_size=batch_size,
                                                  group_size=30),
            steps_per_epoch=steps_per_epoch,
            validation_data=MultimodalDataset.flow_image_from_dir(root=test_root, max_frames_per_video=450,
                                                                  batch_size=batch_size,
                                                                  group_size=group_size),
            validation_steps=steps_per_epoch_val,
            epochs=args.nb_epoch_transfer_learn,
            callbacks=[],
            verbose=1)

        self.setup_to_finetune(model)

        checkpointer = ModelCheckpoint(
            filepath='models/vision/lstmconv_'+args.architecture+'_{acc:2f}-{val_acc:.2f}.hdf5',
            verbose=0,
            monitor='val_acc',
            save_best_only=True)

        model.fit_generator(
            MultimodalDataset.flow_image_from_dir(root=train_root, max_frames_per_video=450, batch_size=batch_size,
                                                  group_size=30),
            steps_per_epoch=steps_per_epoch,
            validation_data=MultimodalDataset.flow_image_from_dir(root=test_root, max_frames_per_video=450,
                                                                  batch_size=batch_size,
                                                                  group_size=group_size),
            validation_steps=steps_per_epoch_val,
            epochs=args.nb_epoch_fine_tune,
            callbacks=[checkpointer],
            verbose=1)


    def get_fbf_model(self, args):
        base_model = None

        if args.architecture == "mobilenet":
            base_model = MobileNet(input_shape=(self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS), weights='imagenet',
                                   include_top=False)
            self.NB_LAYERS_TO_FREEZE = 54
        if args.architecture == "resnet":
            base_model = ResNet50(input_shape=(self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS), weights='imagenet',
                                  include_top=False)
            self.NB_LAYERS_TO_FREEZE = 163

        if args.architecture == "inception":
            base_model = InceptionV3(input_shape=(self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS), weights='imagenet',
                                     include_top=False)
            self.NB_LAYERS_TO_FREEZE = 172

        model = self.add_new_last_layer(base_model, args.num_classes, args.fc_size, args.dropout)

        return base_model, model

    def train_fbf(self, args):
        """Use transfer learning and fine-tuning to train a network on a new dataset"""
        nb_train_samples = MultimodalDataset.get_total_size(self.train_root)
        nb_val_samples = MultimodalDataset.get_total_size(self.test_root)
        nb_epoch_fine_tune = int(args.nb_epoch_fine_tune)
        nb_epoch_transferlearn = int(args.nb_epoch_transferlearn)
        batch_size = int(args.batch_size)

        if args.dataset == 'vuzix':
            flow_from_dir = VuzixDataset.flow_images_from_dir
        else:
            flow_from_dir = MultimodalDataset.flow_image_from_dir

        # setup model
        base_model, model = self.get_fbf_model(args)
        # fine-tuning

        self.setup_to_transfer_learn(model, base_model)

        if args.fine_tuned_weights == "" and args.fbf_model_weights == "":

            model.fit_generator(
                flow_from_dir(root=args.train_dir, batch_size=args.batch_size),
                steps_per_epoch=math.ceil(nb_train_samples/batch_size),
                epochs=nb_epoch_transferlearn,
                validation_data=flow_from_dir(root=args.val_dir, batch_size=args.batch_size),
                validation_steps=math.ceil(nb_val_samples/batch_size),
                class_weight='auto')

            model.save_weights('models/vision/'+args.architecture+'_finetune.hdf5')
            # transfer learning

        else:
            if args.fbf_model_weights == "":
                model.load_weights(args.fine_tuned_weights)
            else:
                model.load_weights(args.fbf_model_weights)

        self.setup_to_finetune(model, args.fine_tune_lr)

        checkpointer = ModelCheckpoint(
            filepath='models/vision/'+args.architecture+'.{epoch:03d}-{acc:2f}-{val_acc:.2f}.hdf5',
            verbose=0,
            monitor='val_acc',
            save_best_only=True)

        history_tl = model.fit_generator(
            MultimodalDataset.flow_image_from_dir(args.train_dir, batch_size=batch_size),
            epochs=nb_epoch_fine_tune,
            steps_per_epoch=math.ceil(nb_train_samples/batch_size),
            validation_data=MultimodalDataset.flow_image_from_dir(args.val_dir, batch_size=batch_size),
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
    a.add_argument("--nb_epoch_fine_tune", default=20, type=int)
    a.add_argument("--nb_epoch_transfer_learn", default=10, type=int)
    a.add_argument('--fine_tune_lr', default=0.0001, type=float)
    a.add_argument("--fine_tuned_weights", default="")
    a.add_argument("--fbf_model_weights", default="")
    a.add_argument("--batch_size", default=150, type=int)
    a.add_argument("--plot", action="store_true")
    a.add_argument("--dropout", default=0.6, type=float)
    a.add_argument("--fc_size", default=1024, type=int)
    a.add_argument("--architecture", default="inception")
    a.add_argument("--dataset", default="multimodal", type=str)
    a.add_argument("--limit_resources", default=False)
    a.add_argument("--fps", default=30, type=int)
    a.add_argument("--model_to_train", default="lrcn", type=str)
    a.add_argument("--lstm_size", default=32, type=int)
    a.add_argument("--num_classes", default=20, type=int)

    vision_agent = VisionAgent()
    args = a.parse_args()

    if args.limit_resources:
        config = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4,
                                allow_soft_placement=True, device_count={'CPU': 1}, allow_grouth=True)
        session = tf.Session(config=config)
        K.set_session(session)

    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print("directories do not exist")
        sys.exit(1)

    if args.model_to_train == "fbf":
        vision_agent.train_fbf(args)

    if args.model_to_train == "lrcn":
        vision_agent.train_lrcn(args)