from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
from keras import regularizers
import numpy as np
import softmax

# this is the size of our encoded representations
encoding_dim = 196  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                    activity_regularizer=regularizers.activity_l1(10e-5))


(x_train, y_train), (x_test, y_test) = mnist.load_data()

unlabeled_train_index = np.argwhere(y_train >= 5)
labeled_train_index = np.argwhere(y_train < 5)

num_train = round(labeled_train_index.shape[0] / 2)

train_index = labeled_train_index[0:num_train]
test_index = labeled_train_index[num_train:]

unlabeled_data = x_train[unlabeled_train_index, :]
unlabeled_data = unlabeled_data.astype('float32') / 255.
unlabeled_data = unlabeled_data.reshape((len(unlabeled_data), np.prod(unlabeled_data.shape[1:])))

print x_train.shape
print x_test.shape

autoencoder.fit(unlabeled_data, unlabeled_data,
                nb_epoch=50,
                batch_size=256,
                shuffle=True)

labeled_train = x_train[train_index, :]
labeled_train = labeled_train.astype('float32') / 255.
labeled_train = labeled_train.reshape((len(labeled_train), np.prod(labeled_train.shape[1:])))

labeled_train_y = y_train[train_index]

labeled_test = x_train[test_index, :]
labeled_test = labeled_test.astype('float32') / 255.
labeled_test = labeled_test.reshape((len(labeled_test), np.prod(labeled_test.shape[1:])))
labeled_test_y = y_train[test_index]

train_features = encoder.predict(labeled_train)
test_features = encoder.predict(labeled_test)

print(train_features.shape)
print(test_features.shape)

# lambda_ = 1e-4
# options_ = {'maxiter': 400, 'disp': True}
#
# tr_features = train_features.T
# labeled_train_y = labeled_train_y.reshape(len(labeled_train_y))
# opt_theta, input_size, num_classes = softmax.softmax_train(encoding_dim, 5,
#                                                            lambda_, tr_features,
#                                                            labeled_train_y, options_)
#
# ##======================================================================
# ## STEP 5: Testing
# ts_features = test_features.T
# labeled_test_y = labeled_test_y.reshape(len(labeled_test_y))
# predictions = softmax.softmax_predict((opt_theta, input_size, num_classes), ts_features)
# print("Accuracy: {0:.2f}%".format(100 * np.sum(predictions == labeled_test_y, dtype=np.float64) / labeled_test_y.shape[0]))



softmax = Sequential()
softmax.add(Dense(input_dim=train_features.shape[1], output_dim=5))
softmax.add(Dense(output_dim=5, activation='softmax'))
softmax.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

print("Training Softmax on labeled data...")
softmax.fit(train_features, to_categorical(labeled_train_y),
            nb_epoch=100,
            batch_size=256,
            verbose=2,
            validation_split=0.1)
pred = softmax.predict(test_features)
scores = softmax.evaluate(test_features, to_categorical(labeled_test_y), verbose=0)
print("Accuracy: {}".format(scores[1] * 100))
# print("Accuracy: {0:.2f}%".format(100 * np.sum(np.argmax(pred, axis=1) == labeled_test_y.reshape(len(labeled_test_y)),
#                                                dtype=np.float64) / labeled_test_y.shape[0]))

