import tensorflow as tf
import logging
import sys, signal
import time
from arguments import args
import os
from SensorAgent import SensorAgent
from VisionAgent import VisionAgent
from ActivityEnvironment import ActivityEnvironment
from a3c import A3C
import datetime
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Disables write_meta_graph argument, which freezes entire process and is
# mostly useless.
class FastSaver(tf.train.Saver):
    def save(self, sess, save_path, global_step=None, latest_filename=None,
             meta_graph_suffix='meta', write_meta_graph=True):
        super(FastSaver, self).save(sess, save_path, global_step,
                                    latest_filename, meta_graph_suffix, False)

def run(args, server):
    # env = create_env(args.env_id, client_id=str(args.task),
    #                  remotes=args.remotes)
    sensor_agent = SensorAgent()
    vision_agent = VisionAgent()
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")

    if args.dttype == 'multimodal':
        dataset_type = args.dttype
        img_chunk_size = 10
        sns_chunk_size = 10
        max_samples = 150
        iter_copy_over = 5

    if args.dttype == 'vuzix':
        dataset_type = args.dttype
        img_chunk_size = 15
        sns_chunk_size = 15
        max_samples = 4500
        iter_copy_over = 20

    env = ActivityEnvironment(dataset=args.dataset,
                              sensor_agent=sensor_agent, vision_agent=vision_agent, alpha=args.alpha, env_id=args.task,
                              time=current_time,img_chunk_size=img_chunk_size, sns_chunk_size=sns_chunk_size,
                              dt_type=dataset_type, img_max_samples=max_samples, sns_max_samples=max_samples)

    trainer = A3C(env, args.task, args.visualise, args.sensorpb, args.visionpb)

    # Variable names that start with 'local' are not saved in checkpoints.
    variables_to_save = [v for v in tf.global_variables()
                         if not v.name.startswith('local')]
    init_op = tf.variables_initializer(variables_to_save)
    ready_op = tf.report_uninitialized_variables(variables_to_save)
    init_all_op = tf.global_variables_initializer()
    saver = FastSaver(variables_to_save)

    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 tf.get_variable_scope().name)
    logger.info("Trainable vars:")
    for v in var_list:
        logger.info("  {} {}".format(v.name, v.get_shape()))

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)

    config = tf.ConfigProto(
            device_filters=['/job:ps',
                            '/job:worker/task:{}/cpu:0'.format(args.task)])
    logdir = os.path.join(args.log_dir, 'train')

    summary_writer = tf.summary.FileWriter("{} + {}".format(logdir,args.task))

    logger.info("Events directory: {}_{}".format(logdir, args.task))

    if args.evaluate:
        save_model_secs = 0
        save_summaries_secs = 0
    else:
        save_model_secs = 30
        save_summaries_secs = 30

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                             logdir=logdir,
                             saver=saver,
                             summary_op=None,
                             init_op=init_op,
                             init_fn=init_fn,
                             summary_writer=summary_writer,
                             ready_op=ready_op,
                             global_step=trainer.global_step,
                             save_model_secs=save_model_secs, #TODO: move to arguments
                             save_summaries_secs=save_summaries_secs) #TODO: move to arguments

    num_global_steps = 100000000 # TODO: move to arguments

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to "
        "connect to the parameter server. One common cause is that the "
        "parameter server DNS name isn't resolving yet, or is misspecified.")
    if not args.evaluate:
        with sv.managed_session(server.target,
                                config=config) as sess, sess.as_default():
            sess.run(trainer.sync)
            trainer.start(sess, summary_writer, iter_copy_over)
            global_step = sess.run(trainer.global_step)
            logger.info("Starting training at step={}".format(global_step))
            while not sv.should_stop() and (not num_global_steps or
                                            global_step < num_global_steps):
                trainer.process(sess)
                global_step = sess.run(trainer.global_step)
    else:
        with sv.managed_session(server.target, config=config) as sess, sess.as_default():
            sess.run(trainer.sync)
            global_step = sess.run(trainer.global_step)
            logger.info("Starting training at step={}".format(global_step))
            episode_generator = env.episode_generator(stop_at_full_sweep=True)
            generator = env.sample_from_episode(episode_generator)
            y_arr, y_true_arr, actions = [], [], []
            while not sv.should_stop():
                try:
                    y, y_true, action = trainer.evaluate(generator, env, sess)
                    y_arr.append(y)
                    y_true_arr.append(y_true)
                    actions.append(action)
                except StopIteration:
                    stacked = np.column_stack((y_arr, y_true_arr, actions))
                    accuracy = sum(stacked[:, 0] == stacked[:, 1])/float(len(y_true_arr))
                    np.save(str(round(accuracy, 2))+'_policy_results.npy', stacked)
                    print("Final Accuracy {}".format(accuracy))
                    sv.request_stop()

    # Ask for all the services to stop.
    sv.stop()
    logger.info("reached {} steps. worker stopped.".format(global_step))

def cluster_spec(num_workers, num_ps):
    """
    More tensorflow setup for data parallelism
    """
    cluster = {}
    port = 12222

    all_ps = []
    host = '127.0.0.1'
    for _ in range(num_ps):
        all_ps.append("{}:{}".format(host, port))
        port += 1
    cluster['ps'] = all_ps

    all_workers = []
    for _ in range(num_workers):
        all_workers.append("{}:{}".format(host, port))
        port += 1
    cluster['worker'] = all_workers
    return cluster

def main(_):
    """
    Setting up Tensorflow for data parallel work
    """

    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn("Received signal {}: exiting".format(signal))
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.job_name == 'worker':
        config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=2)

        server = tf.train.Server(cluster, job_name='worker',
                                 task_index=args.task,
                                 config=config)
        run(args, server)
    else:
        config = tf.ConfigProto(device_filters=['/job:ps'])
        server = tf.train.Server(cluster, job_name='ps', task_index=args.task,
                                 config=config)
        while True:
            time.sleep(1000)

if __name__ == '__main__':
    tf.app.run()
