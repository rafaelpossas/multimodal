from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.layers.wrappers import *
from keras import backend as K
from MultimodalDataset import MultimodalDataset
from VisionAgent import VisionAgent
import numpy as np
from keras.applications.inception_v3 import InceptionV3
import cv2
import glob
import math
from scipy.stats import mode
import argparse
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

def intermediate_model(weights = "models/vision/inception.019-0.867948-0.78.hdf5") :
    def pop(model):
        """Removes the last layer in the model.
        # Raises
            TypeError: if there are no layers in the model.
        """
        if not model.layers:
            raise TypeError('There are no layers in the model.')

        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
            # update self.inbound_nodes
            model.inbound_nodes[0].output_tensors = model.outputs
            model.inbound_nodes[0].output_shapes = [model.outputs[0]._keras_shape]
        model.built = False

    _, model = VisionAgent().get_model(fc_size=1024, architecture='inception')
    model.load_weights(weights)
    # get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                                   [model.layers[-2].output])
    #model = Model(model.input, model.layers[-3].output)
    return model



NB_IV3_LAYERS_TO_FREEZE = 54

def model() :
    K.set_learning_phase(0)
    base_model = intermediate_model()
    base_model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    base_model.trainable = False
    x = Input(shape=(None, 224, 224, 3))
    td_base = TimeDistributed(base_model)(x)
    lstm = LSTM(1024, return_sequences=False, input_shape=(30, 2048))(td_base)
    # x = Flatten()(x)
    # x = Dense(512, activation='relu')(x)
    #dropout = Dropout(0.3)(lstm)
    y = Dense(20, activation='softmax')(lstm)
    model = Model(inputs=[x], outputs=y)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()

    return model

def flow_image_from_dir(root, max_frames_per_video=450, batch_size=10, group_size=1, im_model=None):
    x = []
    y = []

    activity_dict = {
        'act01': (0, 'walking'), 'act02': (1, 'walking upstairs'), 'act03': (2, 'walking downstairs'),
        'act04': (3, 'riding elevator up'), 'act05': (4, 'riding elevator down'),
        'act06': (5, 'riding escalator up'),
        'act07': (6, 'riding escalator down'), 'act08': (7, 'sitting'), 'act09': (8, 'eating'),
        'act10': (9, 'drinking'),
        'act11': (10, 'texting'), 'act12': (11, 'phone calls'), 'act13': (12, 'working on pc'),
        'act14': (13, 'reading'),
        'act15': (14, 'writing sentences'), 'act16': (15, 'organizing files'), 'act17': (16, 'running'),
        'act18': (17, 'push-ups'),
        'act19': (18, 'sit-ups'), 'act20': (19, 'cycling')

    }
    while True:
        files = sorted(glob.glob(os.path.join(root, '*', '*', '*.jpg')))
        all_grouped_files = list()
        if group_size > 1:
            cur_activity = ""
            cur_seq = ""

            for img_file in files:

                img_file_split = img_file.split(os.path.sep)
                cur_ix = int(img_file_split[-1].split(".")[0].split("_")[-1])

                if cur_activity != img_file_split[-3] or cur_seq != img_file_split[-2]:
                    cur_activity = img_file_split[-3]
                    cur_seq = img_file_split[-2]
                    grouped_files = list()

                if len(grouped_files) < group_size and cur_ix <= max_frames_per_video:

                    grouped_files.append(img_file)
                    cur_ix += 1

                if len(grouped_files) == group_size:
                    all_grouped_files.append(grouped_files)
                    grouped_files = []

        files = all_grouped_files if len(all_grouped_files) > 0 else files
        np.random.shuffle(files)

        for img_ix, img in enumerate(files):
            cur_img_batch = []
            if img_ix < max_frames_per_video:
                if type(img) is not list:
                    img = [img]

                activity = img[0].split(os.path.sep)[-3]

                for img_file in img:

                    cur_img = cv2.resize(cv2.imread(img_file), (224, 224)).astype('float')

                    cur_img /= 255.
                    cur_img -= 0.5
                    cur_img *= 2.
                    cur_img = im_model.predict(cur_img[np.newaxis])
                    cur_img_batch.append(cur_img)

                x.append(np.squeeze(cur_img_batch))

                y.append(activity_dict[activity][0])

                if len(x) == batch_size:
                    #print(img)
                    yield np.array(x), np.eye(20)[np.array(y).astype(int)]
                    x, y = ([], [])
def setup_to_finetune(model, fine_tune_lr=0.0001):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=fine_tune_lr, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir", default='multimodal_dataset/video/images/train')
    a.add_argument("--val_dir", default='multimodal_dataset/video/images/test')
    a.add_argument("--batch_size", default=1, type=int)
    a.add_argument("--group_size", default=30, type=int)
    a.add_argument("--nb_epoch_fine_tune", default=20, type=int)
    a.add_argument("--nb_epoch_transferlearn", default=10, type=int)
    args = a.parse_args()

    train_root = args.train_dir
    test_root = args.val_dir
    group_size = args.group_size
    batch_size = args.batch_size

    total_train_size = MultimodalDataset.get_total_size(train_root)
    total_test_size = MultimodalDataset.get_total_size(test_root)
    steps_per_epoch = math.ceil((total_train_size / group_size))
    steps_per_epoch_val = math.ceil((total_test_size / group_size))

    #base_model = MobileNet(input_shape=(group_size, 224, 224, 3), include_top=False, pooling="avg")
    # model = model()
    #
    checkpointer = ModelCheckpoint(
        filepath='models/vision/lstmconv_{acc:2f}-{val_acc:.2f}.hdf5',
        verbose=0,
        monitor='val_acc',
        save_best_only=True)

    #setup_to_transfer_learn(model, base_model)
    # x = []
    # y = []
    #
    # for ix in range(0, steps_per_epoch_val):
    #     print(ix)
    #     x_batch, y_batch = next(flow_image_from_dir(root=test_root, max_frames_per_video=450, batch_size=1,
    #                                                group_size=group_size, im_model=im_model))
    #     x.append(np.squeeze(x_batch))
    #     y.append(np.squeeze(y_batch))
    #
    # np.save('test_features_x.npy',x)
    # np.save("test_features_y.npy",y)
    # x = np.load("features_x.npy")
    # y = np.load("features_y.npy")
    # x_test = np.load("test_features_x.npy")
    # y_test = np.load("test_features_y.npy")

    model = model()
    model.fit_generator(
        MultimodalDataset.flow_image_from_dir(root=train_root, max_frames_per_video=450, batch_size=batch_size,
                                              group_size=30),
        steps_per_epoch=steps_per_epoch,
        validation_data= MultimodalDataset.flow_image_from_dir(root=test_root, max_frames_per_video=450,
                                                               batch_size=batch_size,
                                                               group_size=group_size),
        validation_steps=steps_per_epoch_val,
        epochs=args.nb_epoch_transferlearn,
        callbacks=[checkpointer],
        verbose=1)

    # checkpointer = ModelCheckpoint(
    #     filepath='models/vision/lstmconv.{epoch:03d}-{acc:2f}-{val_acc:.2f}.hdf5',
    #     verbose=0,
    #     monitor='val_acc',
    #     save_best_only=True)
    #
    # setup_to_finetune(model)
    #
    # model.fit_generator(
    #     MultimodalDataset.flow_image_from_dir(root=train_root, max_frames_per_video=450, batch_size=batch_size,
    #                                           group_size=group_size),
    #     steps_per_epoch=steps_per_epoch,
    #     validation_data= MultimodalDataset.flow_image_from_dir(root=test_root, max_frames_per_video=450, batch_size=batch_size,
    #                                                            group_size=group_size),
    #     validation_steps=steps_per_epoch_val,
    #     epochs=args.nb_epoch_fine_tune,
    #     callbacks=[checkpointer],
    #     verbose=1)
