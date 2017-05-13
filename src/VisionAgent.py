from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.datasets import cifar100
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
input_tensor = Input(shape=(32, 32, 3))
y_train = np.eye(len(np.unique(y_train)))[y_train.reshape(len(y_train))]

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
predictions = Dense(100, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()
for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(x_train, y_train)