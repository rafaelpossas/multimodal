from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM

import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np
from src.MultimodalDataset import MultimodalDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 21760, 'size of training batches')
tf.app.flags.DEFINE_integer('epochs', 5000, 'number of training iterations')
tf.app.flags.DEFINE_integer('timesteps_input', 15, 'number of steps for the input tensor')
tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/rnn.ckpt', 'path to checkpoint file')
tf.app.flags.DEFINE_string('train_data', 'data/mnist_train.csv', 'path to train and test data')
tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')


class SaveAtEnd(tf.train.SessionRunHook):

    _saver = None

    def begin(self):
        self._saver = tf.train.Saver()

    def end(self, session):
        self._saver.save(session, FLAGS.checkpoint_file_path)

class RNN:

    def __init__(self, timesteps_input=15, output_size=20, layers=[100], learning_rate=0.001):
        """
        :param timesteps_pred: the number of timesteps to predict in the future
        :param look_back: how many timesteps to look back
        :param features: number of dimensions (i.e. payments)
        """
        self._timesteps_input = timesteps_input
        self._output_size = output_size
        self._layers = layers
        self._learning_rate = learning_rate
        self._batch_size = FLAGS.batch_size



    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def _inference(self, X):
        with tf.variable_scope('output_layer'):
            cell = tf.contrib.rnn.BasicRNNCell(num_units=self._layers[0], activation=tf.nn.relu)
            rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

            logits = tf.contrib.layers.fully_connected(states, self._output_size, activation_fn=None)

        return logits, tf.nn.softmax(logits)

    def _loss_fn(self, y, logits):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
            cost = tf.reduce_mean(cross_entropy, name=scope.name)
            accuracy = tf.metrics.accuracy(labels=tf.argmax(y, 1), predictions=tf.arg_max(logits, 1))
            tf.summary.scalar('cost', cost)
        return cost, accuracy

    def _optimize(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def predict(self, X):
        logits, softmax = self._inference(X)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, FLAGS.checkpoint_file_path)
            pred = softmax.eval(session=sess)

        return pred

    def train(self, input_x, output_y, epochs=None):
        counter = 0
        epoch = 1

        dataset = tf.contrib.data.Dataset.from_tensor_slices((input_x, output_y))
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.repeat(FLAGS.epochs)

        loss_per_epoch = []
        accuracy_per_epoch = []

        with tf.Graph().as_default():

            x = tf.placeholder(shape=[None, self._timesteps_input, 6], dtype=tf.float32, name='input')
            y = tf.placeholder(shape=[None, self._output_size], dtype=tf.float32, name='output')

            global_step = tf.contrib.framework.get_or_create_global_step()

            logits, softmax = self._inference(x)
            loss, accuracy = self._loss_fn(y, logits)

            summary_op = tf.summary.merge_all()
            train_op = self._optimize(loss, global_step=global_step)

            init = tf.global_variables_initializer()
            #saver = tf.train.Saver()

            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()

            checkpoint_saver_hook = SaveAtEnd()
            training_session = tf.train.MonitoredTrainingSession(hooks=[checkpoint_saver_hook])

            with training_session as sess:
                writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
                sess.run(init)
                while not sess.should_stop():
                    if counter != 0 and counter % math.ceil(input_x.shape[0] / self._batch_size) == 0:
                        if epoch % 10 == 0:
                            print("Epoch {} running on batch size of {} loss {} accuracy {}"
                                  .format(epoch, self._batch_size, np.mean(loss_per_epoch), np.mean(accuracy_per_epoch)))
                        epoch += 1
                        loss_per_epoch = []

                    x_batch, y_batch = sess.run(next_element)
                    _, epoch_loss, acc, summary = sess.run([train_op, loss, accuracy, summary_op], feed_dict={x: x_batch,
                                                                                               y: y_batch})
                    accuracy_per_epoch.append(acc)
                    loss_per_epoch.append(epoch_loss)
                    writer.add_summary(summary, counter)
                    counter += 1

                    # if epoch == FLAGS.epochs - 1:
                    #     saver.save(sess, FLAGS.checkpoint_file_path, global_step)


def load_sensor_dataset():

    multimodal_dataset = MultimodalDataset()

    train_sns_x, train_sns_y = multimodal_dataset.load_all_sensor_files(selected_sensors=['accx','accy','accz','gyrx','gyry','gyrz'],
                                                                        sensor_root='multimodal_dataset/sensor/train')

    train_onehot_y = np.eye(20)[np.squeeze(train_sns_y).astype(int)]

    train_sns_x, train_onehot_y = multimodal_dataset.split_windows(FLAGS.timesteps_input, 1, train_sns_x, train_onehot_y)

    print(train_sns_x.shape)

    test_sns_x, test_sns_y =  multimodal_dataset.load_all_sensor_files(selected_sensors=['accx','accy','accz','gyrx','gyry','gyrz'],
                                                                        sensor_root='multimodal_dataset/sensor/test')
    test_onehot_y = np.eye(20)[np.squeeze(test_sns_y).astype(int)]

    test_sns_x, test_onehot_y = multimodal_dataset.split_windows(FLAGS.timesteps_input, 1, test_sns_x, test_onehot_y)

    print(test_sns_x.shape)

    return train_sns_x, train_onehot_y, test_sns_x, test_onehot_y

if __name__ == '__main__':

    rnn = RNN()
    #
    # def generate_data():
    #     t_min, t_max = 0, 30
    #     resolution = 0.1
    #
    #     def time_series(t):
    #         return t * np.sin(t) / 3 + 2 * np.sin(t * 5)
    #
    #     def next_batch(batch_size, n_steps):
    #         t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    #         Ts = t0 + np.arange(0., n_steps + 1) * resolution
    #         ys = time_series(Ts)
    #         return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
    #
    #     surrogate_x = []
    #     surrogate_y = []
    #
    #     for _ in range(100):
    #         X_batch, y_batch = next_batch(50, FLAGS.timesteps_input)
    #         surrogate_x.append(X_batch)
    #         surrogate_y.append(y_batch)
    #
    #     surrogate_x = np.array(surrogate_x).reshape((-1, FLAGS.timesteps_input, 1))
    #     surrogate_y = np.array(surrogate_y).reshape((-1, FLAGS.timesteps_output, 1))
    #
    #     return surrogate_x, surrogate_y
    x, y, x_test, y_test = load_sensor_dataset()
    rnn.train(x, y)
    # pred = rnn.predict(input)
    # plt.plot(pred.flatten())
    # plt.plot(y[0:10].flatten())
    # plt.show()