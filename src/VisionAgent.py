from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
import numpy as np
import h5py
import random
f = h5py.File("activity_frames.hdf5")


def multimodal_generator(file, batch_size):
    while True:
        total_size = file['x'].shape[0]
        index = random.sample(range(0, total_size), batch_size)
        x = file['x'][sorted(index)]
        y = file['y'][sorted(index)]
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        y = np.eye(20)[np.repeat(y, batch_size)]
        yield x, y

input_tensor = Input(shape=(224, 224, 3))

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(20, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit_generator(multimodal_generator(f, batch_size=10), steps_per_epoch=10, epochs=100, verbose=1)