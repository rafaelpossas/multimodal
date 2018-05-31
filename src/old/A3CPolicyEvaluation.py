import tensorflow as tf

class A3CPolicyEvaluation():

    def __init__(self):
        return None

    def evaluate(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "/root/dev/final/ego0/train/model.ckpt-113825")


if __name__=="__main__":
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/root/dev/final/ego0/train/model.ckpt-113825")