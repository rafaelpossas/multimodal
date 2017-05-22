if __name__=='__main__':

    latent_dim = 100
    input_dim = 3
    timesteps = 150
    step_size = 1
    nb_epoch = 10
    batch_size = 6000
    verbose = 1

    dt = SensorDatasetUCI("/root/dev/multimodal/uci_cleaned")
    # scaler = MinMaxScaler()
    # lstm = RegressionLSTM(scaler)
    dt.load_dataset(train_size=0.9, group_size=timesteps, step_size=step_size)
    lstm_ae = AutoencoderLSTM(latent_dim=latent_dim, input_dim=input_dim, timesteps=timesteps)

    early_stopping_ae = EarlyStopping(monitor="loss", min_delta=0.1, patience=5)
    autoencoder = lstm_ae.get_stacked_autoencoder_model()
    #autoencoder.load_weights("./src/models/pre_trained_ae_3_axes.hdf5")
    lstm_ae.fit_ae(x_train=dt.x_train, model=autoencoder, save_model=False,
                   nb_epoch=100, batch_size=batch_size,
                   callbacks=[early_stopping_ae])
    autoencoder.save_weights('./src/models/ae_3_layers_stacked.hdf5')
    autoencoder.layers[0].layers[0].get_weights()
    early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.01, patience=10)

    print("\n\nPre-trained Model")
    filepath = "./src/models/{val_acc:.2f}_pt_stacked.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)
    stacked_model_pt = lstm_ae.get_stacked_model(classes=dt.y_train.shape[1], pre_trained_model=autoencoder)
    stacked_model_pt.fit(dt.x_test, dt.y_test, nb_epoch=100, batch_size=nb_epoch, verbose=2,
                         validation_split=0.1, callbacks=[early_stopping,checkpointer])


    print("\n\nNew Model")
    filepath = "./src/models/{val_acc:.2f}_stacked.hdf5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)
    stacked_model = lstm_ae.get_stacked_model(classes=dt.y_train.shape[1])
    stacked_model.fit(dt.x_test, dt.y_test, nb_epoch=100, batch_size=nb_epoch, verbose=2,
                      validation_split=0.1, callbacks=[early_stopping, checkpointer])




    #x_train, x_test, y_train, y_test = lstm.format_data(dt)
    # train_prediction, test_prediction = lstm.fit_transform(lstm.get_model(), x_train, y_train, x_test,
    #                                                        nb_epoch=10,
    #                                                        batch_size=100,
    #                                                        verbose=0)
    # plot_predictions([dt], [train_prediction], [test_prediction],
    #                         [y_train], [y_test], ['accx'], scaler)


    # print("\n\nNon Pre-trained Model")
    # sensor_model = Sequential()
    # sensor_model.add(LSTM(output_dim=latent_dim, input_shape=(timesteps, input_dim), return_sequences=True))
    # sensor_model.add(LSTM(output_dim=int(latent_dim/2), input_shape=(latent_dim, input_dim)))
    # sensor_model.add(Dense(dt.y_train.shape[1], input_dim=latent_dim, init='zero', activation='softmax'))
    # sensor_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # sensor_model.fit(dt.x_train, dt.y_train, nb_epoch=1, batch_size=20, verbose=verbose)

    # sample = dt.x_test[1, :, :]
    # sample = sample.reshape(1, sample.shape[0], sample.shape[1])
    # code = encoder.predict(sample)
    # code = np.repeat(code, timesteps)
    # code = code.reshape(1, timesteps, latent_dim)
    # reconstructed = decoder.predict(code)
    #
    # plt.plot(sample[0])
    # plt.plot(reconstructed[0])


    # inputs = Input(shape=(timesteps, input_dim))
    # encoded = LSTM(latent_dim)(inputs)
    #
    # decoded = RepeatVector(timesteps)(encoded)
    # decoded = LSTM(input_dim, return_sequences=True)(decoded)
    #
    # sequence_autoencoder = Model(inputs, decoded)
    # encoder = Model(inputs, encoded)
    #
    # encoded_input = Input(shape=(timesteps,latent_dim))
    # layer = sequence_autoencoder.layers[-1]
    # decoder = Model(input=encoded_input, output=layer(encoded_input))
    #
    # sequence_autoencoder.compile(optimizer='adadelta', loss="mean_squared_error")
    # sequence_autoencoder.fit(dt.x_train, dt.x_train, nb_epoch=20, batch_size=20)
    #
    # sample = dt.x_test[0, :, :]
    # encoded_seq = encoder.predict(sample.reshape(1, sample.shape[0], sample.shape[1]))
    # encoded_seq = np.repeat(encoded_seq, timesteps)
    # decoded_seq = decoder.predict(encoded_seq.reshape(1, timesteps, latent_dim))


    # best_accuracy = 60
    # sensor_columns = [['accx', 'accy', 'accz'],
    #                   ['grax', 'gray', 'graz'],
    #                   ['gyrx', 'gyry', 'gyrz'],
    #                   ['lacx', 'lacy', 'lacz'],
    #                   ['magx', 'magy', 'magz'],
    #                   ['rotx', 'roty', 'rotz', 'rote']]
    # sensors = [x for l in range(1, len(sensor_columns)) for x in combinations(sensor_columns, l)]
    # # grid = dict(optimizers=['rmsprop', 'adagrad', 'adam','adadelta'],
    # #             layer_size=['32', '64', '128', '256'],
    # #             group_size=['10', '30', '50', '75'],
    # #             dropout=['0.2', '0.4', '0.6', '0.8'])
    # grid = dict(optimizers=['rmsprop'],
    #             layer_size=['64'],
    #             group_size=['75'],
    #             dropout=['0.4'])
    # grid_comb = [(x, y, z, w) for x in grid['optimizers'] for y in grid['layer_size'] for z in grid['group_size'] for w in grid['dropout']]
    # lstm = SensorLSTM()
    # scaler = MinMaxScaler()
    # for sensor in sensors:
    #     sensor = [e for l in sensor for e in l]
    #     #Loading Data and creating model
    #     for grd in grid_comb:
    #         print("Current Sensors {}".format(sensor))
    #         dt.load_dataset(selected_sensors=sensor,
    #                         group_size=int(grd[2]), step_size=int(grd[2]), train_size=0.9)
    #
    #         model = lstm.get_model(input_shape=(dt.x_train.shape[1], dt.x_train.shape[2]),
    #                                output_shape=dt.y_train.shape[1], layer_size=int(grd[1]),
    #                                optimizer=grd[0], dropout=float(grd[3]))
    #         #Callbacks
    #         #filepath = "./models/{val_acc:.2f}_"+'_'.join(sensor)+".hdf5"
    #         #checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True)
    #         #reduce_lr_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.01, verbose=1)
    #         early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=30)
    #
    #
    #         #Scores
    #         scores = lstm.fit_transform(model, dt, nb_epoch=1000, callbacks=[early_stopping])
    #         acc = (scores[1] * 100)
    #         print("Accuracy: %.2f%%" % acc)
    #         filepath = "./models/%.2f_" % acc + '_'.join(sensor)+'_'+'_'.join(grd) + ".hdf5"
    #         #if acc >= best_accuracy:
    #         #best_accuracy = acc
    #         model.save_weights(filepath=filepath)
