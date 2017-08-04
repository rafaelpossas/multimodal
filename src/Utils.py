import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler
from src.MultimodalDataset import MultimodalDataset


def plot_predictions(dt_array,
                     array_train_prediction, array_test_prediction,
                     array_y_train, array_y_test, labels=[],
                     scaler=MinMaxScaler()):
    number_axes = len(dt_array)
    f, axarr = plt.subplots(number_axes, sharex=True, figsize=(8, 8))
    i = 0

    for dt, train_prediction, test_prediction, y_train, y_test, label \
            in zip(dt_array, array_train_prediction, array_test_prediction,
                   array_y_train, array_y_test, labels):
        X = np.vstack((dt.x_train[:, :, :], dt.x_test[:, :, :]))
        X = X.reshape([X.shape[0] * X.shape[1], 1])
        X = scaler.fit_transform(X)

        train_prediction = scaler.inverse_transform(train_prediction)
        y_train = scaler.inverse_transform([y_train])

        test_prediction = scaler.inverse_transform(test_prediction)
        y_test = scaler.inverse_transform([y_test])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[0], train_prediction[:, 0]))
        train_score_string = 'Train Score: %.2f RMSE' % (trainScore)
        testScore = math.sqrt(mean_squared_error(y_test[0], test_prediction[:, 0]))
        test_score_string = 'Test Score: %.2f RMSE' % (testScore)

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(X)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[1:len(train_prediction) + 1, :] = train_prediction
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(X)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_prediction) + (1 * 2) + 1:len(X) - 1, :] = test_prediction
        # plot baseline and predictions
        ax = axarr[i] if number_axes > 1 else axarr
        ax.set_title(label + ' - ' + train_score_string + ' - ' + test_score_string)
        ax.plot(scaler.inverse_transform(X))
        ax.plot(trainPredictPlot)
        ax.plot(testPredictPlot)
        i += 1

    plt.show()


def create_dataset(sensors):
    dataset = MultimodalDataset()
    dataset.load_multimodal_dataset(450, 450, '../multimodal_dataset/video/images/train',
                                    sensor_root='../multimodal_dataset/sensor/',
                                    sensors=sensors,
                                    output_file='multimodal_full_train.hdf5')

    dataset.load_multimodal_dataset(450, 450, '../multimodal_dataset/video/images/test',
                                    sensor_root='../multimodal_dataset/sensor/',
                                    sensors=sensors,
                                    output_file='multimodal_full_test.hdf5')

if __name__=='__main__':
    create_dataset(['accx', 'accy', 'accz'])