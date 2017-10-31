from scipy.stats import mode

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.callbacks import *
import keras.backend as K

from MultimodalDataset import MultimodalDataset
from VuzixDataset import VuzixDataset
from keras.utils.training_utils import multi_gpu_model

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
    NUM_EPOCHS = 0
    INIT_LR = 5e-3

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

    def __init__(self, weights=None, architecture="mobilenet", num_classes=20,
                 fc_size=512, dropout=0.6, tf_input=None, tf_output=None):

        if weights is not None:
            _, self.model = self.get_fbf_model(architecture=architecture,
                                               num_classes=num_classes, fc_size=fc_size, dropout=dropout)
            self.model.load_weights(weights)
            #self.model.save("vision_model_full.hdf5")
        self.input_tf, self.output_tf = tf_input, tf_output

    def predict(self, x):
        if self.model is not None:
            if len(x.shape) < 4:
                x = x[np.newaxis]
            num_samples = x.shape[0]
            pred = self.model.predict(x)
            pred = np.argmax(pred, axis=1)
            pred = pred.reshape((int(pred.shape[0] / num_samples), num_samples, 1))
            return [mode(arr.flatten())[0][0] for arr in pred][0]
        else:
            raise Exception("The CNN model needs to be provided")

    def predict_from_tf(self, input, session=None):
        if len(input.shape) < 3:
            input = input[np.newaxis, :, :]

        if session is None:
            session = tf.Session()

        num_samples = input.shape[0]
        pred = []

        for img_x in input:
            if len(img_x.shape) < 4:
                img_x = img_x[np.newaxis]
            pred.append(session.run([self.output_tf], {self.input_tf: img_x}))

        pred = np.argmax(np.array(np.squeeze(pred)), axis=1)
        pred = pred.reshape((int(pred.shape[0] / num_samples), num_samples, 1))

        return [mode(arr.flatten())[0][0] for arr in pred][0]

    def intermediate_lrcn_model(self, args):
        _, model = self.get_fbf_model(args)

        if args.fbf_model_weights != "":
            model.load_weights(args.fbf_model_weights)

        return model

    def get_lrcn_model(self, args):
        K.set_learning_phase(0)

        base_model = self.intermediate_lrcn_model(args)
        base_model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
        base_model.trainable = False

        x = Input(shape=(None, self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS))
        td_base = TimeDistributed(base_model)(x)
        lstm = LSTM(args.lstm_size, input_shape=(10, 1024), return_sequences=True)(td_base)
        lstm = Dropout(args.dropout)(lstm)
        lstm = LSTM(int(args.lstm_size / 2), return_sequences=True)(lstm)
        flat = Flatten()(lstm)
        fc = Dense(units=128)(flat)
        dropout = Dropout(args.dropout)(fc)
        y = Dense(args.num_classes, activation='softmax')(dropout)

        model = Model(inputs=[x], outputs=y)
        optmizer = RMSprop(decay=0.000001)
        model.compile(optimizer=optmizer,
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
            MultimodalDataset.flow_from_dir(root=train_root, max_frames_per_video=450, batch_size=batch_size,
                                            group_size=30),
            steps_per_epoch=steps_per_epoch,
            validation_data=MultimodalDataset.flow_from_dir(root=test_root, max_frames_per_video=450,
                                                            batch_size=batch_size,
                                                            group_size=group_size),
            validation_steps=steps_per_epoch_val,
            epochs=args.nb_epoch_transfer_learn,
            callbacks=[],
            verbose=1)

        self.setup_to_finetune(model)

        checkpointer = ModelCheckpoint(
            filepath='models/vision/lstmconv_' + args.architecture + '_{acc:2f}-{val_acc:.2f}.hdf5',
            verbose=0,
            monitor='val_acc',
            save_best_only=True)

        model.fit_generator(
            MultimodalDataset.flow_from_dir(root=train_root, max_frames_per_video=450, batch_size=batch_size,
                                            group_size=30),
            steps_per_epoch=steps_per_epoch,
            validation_data=MultimodalDataset.flow_from_dir(root=test_root, max_frames_per_video=450,
                                                            batch_size=batch_size,
                                                            group_size=group_size),
            validation_steps=steps_per_epoch_val,
            epochs=args.nb_epoch_fine_tune,
            callbacks=[checkpointer],
            verbose=1)

    def setup_to_transfer_learn(self, model, base_model):
        """Freeze all layers and compile the model"""
        if base_model is not None:
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for layer in base_model.layers[:-2]:
                layer.trainable = False
        opt = SGD(lr=self.INIT_LR, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

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
        model.compile(optimizer=SGD(lr=fine_tune_lr, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['accuracy'])

    def add_new_last_layer(self, base_model, nb_classes, fc_size=64, dropout=0.2):

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(fc_size, activation='relu')(x)
        x = Dropout(dropout)(x)
        predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer

        with tf.device("/cpu:0"):
            model = Model(inputs=base_model.input, outputs=predictions)

        return model

    def get_fbf_model(self, architecture="mobilenet", num_classes=20, fc_size=512, dropout=0.6, pre_trained_model=None):
        base_model = None

        if architecture == "mobilenet":
            base_model = MobileNet(input_shape=(self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS), weights='imagenet',
                                   include_top=False)
            self.NB_LAYERS_TO_FREEZE = 54
        if architecture == "resnet":
            base_model = ResNet50(input_shape=(self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS), weights='imagenet',
                                  include_top=False)
            self.NB_LAYERS_TO_FREEZE = 163

        if architecture == "inception":
            base_model = InceptionV3(input_shape=(self.IM_WIDTH, self.IM_HEIGHT, self.NUM_CHANNELS), weights='imagenet',
                                     include_top=False)
            self.NB_LAYERS_TO_FREEZE = 172

        model = self.add_new_last_layer(base_model, num_classes, fc_size, dropout)

        if pre_trained_model is not None:
            model.load_weights(pre_trained_model)

        return base_model, model



    def train_fbf(self, args):


        def poly_decay(epoch):
            # initialize the maximum number of epochs, base learning rate,
            # and power of the polynomial
            maxEpochs = self.NUM_EPOCHS
            baseLR = self.INIT_LR
            power = 1.0

            # compute the new learning rate based on polynomial decay
            alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

            # return the new learning rate
            return alpha

        if args.dataset == 'vuzix':
            max_frames_per_video = 4500
            flow_from_dir = VuzixDataset.flow_from_dir
        else:
            max_frames_per_video = 150
            flow_from_dir = MultimodalDataset.flow_from_dir

        """Use transfer learning and fine-tuning to train a network on a new dataset"""
        all_val_samples, _ = MultimodalDataset.get_all_files(args.val_dir, 1,
                                                             max_frames_per_video)
        nb_val_samples = len(all_val_samples)

        all_train_samples, _ = MultimodalDataset.get_all_files(args.train_dir, 1,
                                                               max_frames_per_video)
        nb_train_samples = len(all_train_samples)
        nb_epoch_fine_tune = int(args.nb_epoch_fine_tune)
        nb_epoch_transferlearn = int(args.nb_epoch_transfer_learn)
        batch_size = int(args.batch_size)



        # setup model
        base_model, model = self.get_fbf_model(args.architecture, args.num_classes, args.fc_size, args.dropout,
                                               args.pre_trained_model)
        # fine-tuning
        if args.gpus > 1:
            model = multi_gpu_model(model, gpus=args.gpus)

        self.setup_to_transfer_learn(model, base_model)
        checkpointer = ModelCheckpoint(
            filepath='models/vision/'+args.dataset+'_finetune_' + args.architecture + '_{acc:2f}-{val_acc:.2f}.hdf5',
            verbose=0,
            monitor='val_acc',
            save_best_only=True)

        self.NUM_EPOCHS = nb_epoch_transferlearn
        lr_decay = LearningRateScheduler(poly_decay)

        if args.fine_tuned_weights == "" and args.fbf_model_weights == "":

            model.fit_generator(
                flow_from_dir(root=args.train_dir, max_frames_per_video=150, batch_size=args.batch_size),
                steps_per_epoch=math.ceil(nb_train_samples / batch_size),
                epochs=nb_epoch_transferlearn,
                validation_data=flow_from_dir(root=args.val_dir, max_frames_per_video=150, batch_size=args.batch_size),
                validation_steps=math.ceil(nb_val_samples / batch_size),
                callbacks=[checkpointer, lr_decay])
            # transfer learning

        else:
            if args.fbf_model_weights == "":
                model.load_weights(args.fine_tuned_weights)
            else:
                model.load_weights(args.fbf_model_weights)

        self.setup_to_finetune(model, args.fine_tune_lr)

        checkpointer = ModelCheckpoint(
            filepath='models/vision/' + args.dataset+ "_" + args.architecture
                     + '.{epoch:03d}-{acc:2f}-{val_acc:.2f}.hdf5',
            verbose=1,
            monitor='val_acc',
            save_best_only=True)

        self.INIT_LR = args.fine_tune_lr
        self.NUM_EPOCHS = nb_epoch_fine_tune

        lr_decay = LearningRateScheduler(poly_decay)

        history_tl = model.fit_generator(
            flow_from_dir(args.train_dir, max_frames_per_video=150, batch_size=batch_size),
            epochs=nb_epoch_fine_tune,
            steps_per_epoch=math.ceil(nb_train_samples / batch_size),
            validation_data=flow_from_dir(args.val_dir, max_frames_per_video=150, batch_size=batch_size),
            validation_steps=math.ceil(nb_val_samples / batch_size),
            class_weight='auto',
            callbacks=[checkpointer, lr_decay])

        if args.plot:
            self.plot_training(history_tl)

    def evaluate(self, args):

        if args.dataset == 'vuzix':
            max_frames_per_video = 4500
            flow_from_dir = VuzixDataset.flow_from_dir
        else:
            max_frames_per_video = 150
            flow_from_dir = MultimodalDataset.flow_from_dir

        model = load_model(args.pre_trained_model)

        all_val_samples, _ = MultimodalDataset.get_all_files(args.val_dir, 1,
                                                             max_frames_per_video)
        nb_val_samples = len(all_val_samples)

        all_train_samples, _ = MultimodalDataset.get_all_files(args.train_dir, 1,
                                                               max_frames_per_video)
        nb_train_samples = len(all_train_samples)

        print("Evaluating on Test Set")
        result = model.evaluate_generator(flow_from_dir(root=args.val_dir, group_size=1,
                                                        batch_size=args.batch_size,
                                                        max_frames_per_video=max_frames_per_video, type="img",
                                                        shuffle_arrays=False),
                                          steps=math.floor(nb_val_samples / args.batch_size))
        print(result)

        print("Evaluating on Train Set")
        result = model.evaluate_generator(flow_from_dir(root=args.train_dir, group_size=1,
                                                        batch_size=args.batch_size,
                                                        max_frames_per_video=max_frames_per_video,
                                                        type="img", shuffle_arrays=False),
                                          steps=math.floor(nb_train_samples / args.batch_size))
        print(result)

    def plot_training(self, H, dataset, architecture):
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="test_loss")
        plt.plot(N, H["acc"], label="train_acc")
        plt.plot(N, H["val_acc"], label="test_acc")
        plt.title("Status for {} - {}".format(dataset, architecture))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()

        # save the figure
        plt.savefig(dataset+"_"+architecture+'.png')
        plt.close()


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='multimodal_dataset/video/splits/train')
    a.add_argument("--val_dir", default='multimodal_dataset/video/splits/test')
    a.add_argument("--nb_epoch_fine_tune", default=20, type=int)
    a.add_argument("--nb_epoch_transfer_learn", default=3, type=int)
    a.add_argument('--fine_tune_lr', default=0.0001, type=float)
    a.add_argument("--fine_tuned_weights", default="")
    a.add_argument("--fbf_model_weights", default="")
    a.add_argument("--pre_trained_model", default=None)
    a.add_argument("--batch_size", default=150, type=int)
    a.add_argument("--plot", action="store_true")
    a.add_argument("--dropout", default=0.6, type=float)
    a.add_argument("--fc_size", default=512, type=int)
    a.add_argument("--architecture", default="mobilenet")
    a.add_argument("--dataset", default="multimodal", type=str)
    a.add_argument("--limit_resources", default=False)
    a.add_argument("--fps", default=10, type=int)
    a.add_argument("--model_to_train", default="fbf", type=str)
    a.add_argument("--lstm_size", default=16, type=int)
    a.add_argument("--num_classes", default=20, type=int)
    a.add_argument("--evaluate", action="store_true")
    a.add_argument("--gpus", default=1, type=int)

    vision_agent = VisionAgent()
    args = a.parse_args()
    if args.evaluate:
        vision_agent.evaluate(args)
    else:
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
