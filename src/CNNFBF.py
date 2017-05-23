from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import numpy as np
import h5py
import random



class CNNFBF(object):

    def __init__(self, train_file='train_activity_frames.hdf5', test_file='test_activity_frames.hdf5', ):
        self.checkpointer = ModelCheckpoint(
            filepath='/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
            verbose=1,
            save_best_only=True)

        # Helper: Stop when we stop learning.
        self.early_stopper = EarlyStopping(patience=10)

        self.tensorboard = TensorBoard(log_dir='./logs/')

        self.f_train = h5py.File(train_file)
        self.f_test = h5py.File(test_file)
        self.num_frames_per_sample = 15

    def get_model(self, num_classes, weights=None):
        base_model = InceptionV3(include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        if weights is not None:
            model.load_weights(weights)

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_training_model(self, weights='imagenet'):
        # create the base pre-trained model
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

        base_model = InceptionV3(weights=weights, include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(101, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        model.load_weights('checkpoints/ucf101_imagenet_1.28.hdf5')

        pop(model)
        pop(model)
        pop(model)

        x = model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(20, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        return model


    def get_top_layer_model(self, base_model):
        """Used to train just the top layers of the model."""
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        return base_model


    def get_mid_layer_model(self, model):
        """After we fine-tune the dense layers, train deeper."""
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 172 layers and unfreeze the rest:
        for layer in model.layers[:172]:
            layer.trainable = False
        for layer in model.layers[172:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy'])

        return model


    def image_generator(self,file, batch_size):
        while True:
            total_size = file['x_img'].shape[0]
            index = random.sample(range(0, total_size), batch_size)
            x = file['x_img'][sorted(index)]
            y = file['y_img'][sorted(index)]
            num_frames_per_sample = x.shape[1]
            x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
            y = np.eye(20)[np.repeat(y, num_frames_per_sample)]
            x = x.astype("float") / 255.0
            yield x, y


    def fit(self, model, nb_epoch, generator, callbacks=[]):
        model.fit_generator(
            generator(self.f_train, batch_size=50),
            steps_per_epoch=100,
            validation_data=generator(self.f_test, batch_size=10),
            validation_steps=10,
            epochs=nb_epoch,
            callbacks=callbacks)
        return model


    def train_model(self, weights_file):
        model = self.get_training_model()

        if weights_file is None:
            print("Loading network from ImageNet weights.")
            # Get and train the top layers.
            model = self.get_top_layer_model(model)
            model = self.fit(model, 10, self.multimodal_generator)
        else:
            print("Loading saved model: %s." % weights_file)
            model.load_weights(weights_file)

        # Get and train the mid layers.
        model = self.get_mid_layer_model(model)
        model = self.fit(model, 1000, self.multimodal_generator,
                         [self.checkpointer, self.early_stopper, self.tensorboard])
        return model