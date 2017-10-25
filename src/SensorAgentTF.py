import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from MultimodalDataset import MultimodalDataset

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 500, 'size of training batches')
tf.app.flags.DEFINE_integer('epochs', 50, 'number of training iterations')
tf.app.flags.DEFINE_integer('timesteps_input', 10, 'number of steps for the input tensor')
tf.app.flags.DEFINE_integer('timesteps_output', 20, 'number of steps for the output tensor')
tf.app.flags.DEFINE_string('checkpoint_file_path', 'checkpoints/rnn.ckpt', 'path to checkpoint file')
tf.app.flags.DEFINE_string('train_data', 'data/mnist_train.csv', 'path to train and test data')
tf.app.flags.DEFINE_string('summary_dir', 'graphs', 'path to directory for storing summaries')
tf.app.flags.DEFINE_string("train_dir", 'multimodal_dataset/video/splits/train', 'path to training data')
tf.app.flags.DEFINE_string("val_dir", 'multimodal_dataset/video/splits/test', 'path to test data')


class SaveAtEnd(tf.train.SessionRunHook):
    _saver = None

    def begin(self):
        self._saver = tf.train.Saver()

    def end(self, session):
        self._saver.save(session, FLAGS.checkpoint_file_path)


class SensorAgentTF:
    def __init__(self, timesteps_input=10, timesteps_output=20, output_size=1, layers=[100], learning_rate=0.001):
        """
        :param timesteps_pred: the number of timesteps to predict in the future
        :param look_back: how many timesteps to look back
        :param features: number of dimensions (i.e. payments)
        """
        self._timesteps_output = timesteps_output
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

            stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self._layers[0]])
            stacked_outputs = tf.contrib.layers.fully_connected(stacked_rnn_outputs,
                                                                self._output_size, activation_fn=None)

            pred = tf.reshape(stacked_outputs, [-1, self._timesteps_output, self._output_size])

        return pred

    def _loss_fn(self, y, y_pred):
        with tf.variable_scope('loss') as scope:
            cost = tf.reduce_mean(tf.square(y_pred - y), name=scope.name)
            tf.summary.scalar('cost', cost)
        return cost

    def _optimize(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def predict(self, X):
        forward = self._inference(X)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, FLAGS.checkpoint_file_path)
            pred = forward.eval(session=sess)

        return pred

    def train(self, epochs=None, generator=None):
        counter = 0
        epoch = 1

        nb_train_samples = MultimodalDataset.get_total_size(FLAGS.train_dir)
        nb_val_samples = MultimodalDataset.get_total_size(FLAGS.val_dir)

        loss_per_epoch = []

        with tf.Graph().as_default():

            x = tf.placeholder(shape=[None, self._timesteps_input, 6], dtype=tf.float32, name='input')
            y = tf.placeholder(shape=[None, self._timesteps_output, 1], dtype=tf.float32, name='output')

            global_step = tf.contrib.framework.get_or_create_global_step()

            y_pred = self._inference(x)
            loss = self._loss_fn(y, y_pred)

            summary_op = tf.summary.merge_all()
            train_op = self._optimize(loss, global_step=global_step)

            init = tf.global_variables_initializer()
            # saver = tf.train.Saver()


            checkpoint_saver_hook = SaveAtEnd()
            training_session = tf.train.MonitoredTrainingSession(hooks=[checkpoint_saver_hook])

            # with training_session as sess:
            #     writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)
            #     sess.run(init)
            #     while not sess.should_stop():
            #         if counter != 0 and counter % math.ceil(nb_train_samples / self._batch_size) == 0:
            #             if epoch % 10 == 0:
            #                 print("Epoch {} running on batch size of {} loss {}"
            #                       .format(epoch, self._batch_size, np.mean(loss_per_epoch)))
            #             epoch += 1
            #             loss_per_epoch = []
            with training_session as sess:
                x_batch, y_batch = next(generator)
                _, epoch_loss, summary = sess.run([train_op, loss, summary_op], feed_dict={x: x_batch,
                                                                                           y: y_batch[:, :, np.newaxis]})
            loss_per_epoch.append(epoch_loss)
            #writer.add_summary(summary, counter)
            counter += 1

                    # if epoch == FLAGS.epochs - 1:
                    #     saver.save(sess, FLAGS.checkpoint_file_path, global_step)


if __name__ == '__main__':
    rnn = SensorAgentTF()

    rnn.train(generator=MultimodalDataset.flow_from_dir(root=FLAGS.val_dir, max_frames_per_video=150,
                                                        group_size=FLAGS.timesteps_input,
                                                        batch_size=FLAGS.batch_size, type="sns"))

    # pred = rnn.predict(input)
    # plt.plot(pred.flatten())
    # plt.plot(y[0:10].flatten())
    # plt.show()
