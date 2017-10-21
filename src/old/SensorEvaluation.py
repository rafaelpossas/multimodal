# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:34:29 2017

@author: rafaelpossas
"""

import matplotlib as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from old.LSTM import SensorLSTM
from old.SensorDataset import SensorDataset


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

dt = SensorDataset("/Users/rafaelpossas/Dev/multimodal/sensor")
best_accuracy = 60
lstm = SensorLSTM()
scaler = MinMaxScaler()


sensor_columns = [['accx', 'accy', 'accz'],
                  ['grax', 'gray', 'graz'],
                  ['gyrx', 'gyry', 'gyrz'],
                  ['lacx', 'lacy', 'lacz'],
                  ['magx', 'magy', 'magz'],
                  ['rotx', 'roty', 'rotz', 'rote']]

targets = ['walking', 'walking upstairs', 'walking downstairs', 'rid.elevator up',
                    'rid.elevator down', 'rid.escalator up', 'rid.escalator down', 'sitting',
                    'eating', 'drinking', 'texting', 'mak.phone calls',
                    'working at PC', 'reading', 'writting sentences', 'organizing files',
                    'running', 'doing push-ups', 'doing sit-ups', 'cycling']

sensor = ['accx','accy','accz']
grd = ['rmsprop', '64', '75', '0.4']


dt.load_dataset(selected_sensors=sensor,
                group_size=int(grd[2]), step_size=int(grd[2]), train_size=0.9)



model = lstm.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]),
                       output_shape=dt.y_train.shape[1], layer_size=int(grd[1]),
                       optimizer=grd[0], dropout=float(grd[3]))

model.fit(dt.x_train, dt.y_train, nb_epoch=25, batch_size=20)
#model.load_weights("./src/models/65.00_accx_accy_accz_rmsprop_64_75_0.4.hdf5")
#Callbacks
#filepath = "./models/{val_acc:.2f}_"+'_'.join(sensor)+".hdf5"
#checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)
#reduce_lr_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.01, verbose=1)
#early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=30)


#Scores
#y_pred = model.predict_classes(dt.)
#scores = model.evaluate(dt.x_test, dt.y_test, verbose=0)
#acc = (scores[1] * 100)
#print("Accuracy: %.2f%%" % acc)

y_pred = model.predict_classes(dt.x_test)
rp = classification_report(np.argmax(dt.y_test, axis=1), y_pred, target_names=targets)
cm = confusion_matrix(np.argmax(dt.y_test, axis=1), y_pred)
