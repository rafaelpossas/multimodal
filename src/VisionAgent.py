from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import numpy as np
import h5py
import random

tensorboard = TensorBoard(log_dir='./logs/')

f_train = h5py.File("train_activity_frames.hdf5")
f_test = h5py.File("test_activity_frames.hdf5")
num_frames_per_sample = 10


checkpointer = ModelCheckpoint(
    filepath='/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)


def get_model(weights='imagenet'):
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


def get_top_layer_model(base_model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return base_model


def get_mid_layer_model(model):
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


def multimodal_generator(file, batch_size):
    while True:
        total_size = file['x'].shape[0]
        index = random.sample(range(0, total_size), batch_size)
        x = file['x'][sorted(index)]
        y = file['y'][sorted(index)]
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        y = np.eye(20)[np.repeat(y, num_frames_per_sample)]
        x = x.astype("float") / 255.0
        yield x, y


def train_model(model, nb_epoch, generator, callbacks=[]):
    model.fit_generator(
        generator(f_train, batch_size=50),
        steps_per_epoch=100,
        validation_data=generator(f_test, batch_size=10),
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model


def main(weights_file):
    model = get_model()

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = get_top_layer_model(model)
        model = train_model(model, 10, multimodal_generator)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    model = train_model(model, 1000, multimodal_generator,
                        [checkpointer, early_stopper, tensorboard])
    return model

if __name__ == '__main__':
    weights_file = None
    main(weights_file)