import numpy as np
from MultimodalDataset import MultimodalDataset
from globals import activity_dict
import queue as q
from glob import glob
import os
import h5py
import tensorflow as tf

class ActivityEnvironment(object):

    SENSOR = 0
    CAMERA = 1
    CURRENT_ACTION = SENSOR

    battery_size = 1
    sensor_consumption_per_hour = 30
    camera_consumption_per_hour = 135
    num_img_per_activity = 450
    num_sns_per_activity = 150
    total_seconds = 15

    current_x_activity_sns_buffer = q.Queue()
    current_x_activity_img_buffer = q.Queue()

    current_y_activity_sns_buffer = q.Queue()
    current_y_activity_img_buffer = q.Queue()

    reward_right_pred = 1
    reward_wrong_pred = -1

    current_consumption = 0

    total_steps = 0

    def __init__(self, sensor_agent = None, vision_agent=None,
                 img_chunk_size=10, sns_chunk_size=10,
                 img_root="multimodal_dataset/video/images/",
                 sns_root="multimodal_dataset/sensor/",
                 sensors=['accx', 'accy', 'accz', 'gyrx','gyry','gyrz'],
                 img_max_samples=150, sns_max_samples=150,
                 alpha=0, logger=None, time=None, env_id=0):

        self.img_root = img_root
        self.sns_root = sns_root
        self.sensors = sensors
        self.img_max_samples = img_max_samples
        self.sns_max_samples = sns_max_samples
        self.sensor_agent = sensor_agent
        self.vision_agent = vision_agent
        self.img_chunk_size = img_chunk_size
        self.sns_chunk_size = sns_chunk_size

        self.cur_img_input = None
        self.cur_sns_input = None
        self.cur_img_label = None
        self.cur_sns_label = None

        self.done = False

        self.sensor_consumption_per_step = (self.sensor_consumption_per_hour/3600) * \
                                           ((self.sns_chunk_size*self.total_seconds)/sns_max_samples)
        self.vision_consumption_per_step = (self.camera_consumption_per_hour/3600) * \
                                           ((self.img_chunk_size*self.total_seconds)/img_max_samples)
        self.alpha = alpha
        self.state_generator = self.sample_from_episode()
        self.episode_generator = self._episode_generator()

        self.datasets_full_sweeps = 0
        self.observation_shape = (10, 6)
        self.action_space = 2

        self.all_scores = []
        self.all_rewards = []
        self.all_true_preds = []
        self.all_action_avg = []
        self.moving_average = []
        self.all_acc = []

        self.ep_rewards = []
        self.ep_steps = []
        self.ep_probs = []
        self.ep_true_preds = []

        self.env_id = str(env_id)
        self.logger = logger
        self.current_time = time
        self.info = dict()

    def reset(self):
        self.info = dict()
        self.current_consumption = 0
        self.done = False
        self.done, self.cur_img_input, self.cur_img_label, self.cur_sns_input, self.cur_sns_label = next(self.state_generator)
        self.all_scores = []
        self.all_rewards = []
        self.all_true_preds = []
        self.all_action_avg = []
        self.all_acc = []
        self.moving_average = []
        return self.cur_sns_input

    def calculate_reward(self, real, pred_sns, pred_img, sensor_type):

        real = np.argmax(real)

        if sensor_type == self.SENSOR:
            # When Sensor is Right
            if pred_sns == real and pred_img != real:
                total_reward = self.reward_right_pred

            if pred_sns == real and pred_img == real:
                total_reward = self.reward_right_pred

            # When Sensor is Wrong
            if pred_sns != real and pred_img == real:
                total_reward = self.reward_wrong_pred

            if pred_sns != real and pred_img != real:
                total_reward = self.reward_wrong_pred

        if sensor_type == self.CAMERA:
            # When Camera is Right
            if pred_img == real and pred_sns != real:
                total_reward = self.alpha

            if pred_img == real and pred_sns == real:
                total_reward = self.alpha

            # When Camera is Wrong
            if pred_img != real and pred_sns == real:
                total_reward = self.reward_wrong_pred

            if pred_sns != real and pred_img != real:
                total_reward = self.reward_wrong_pred

        return total_reward


    def step(self, action,episode=0, action_probs=None, session=None, summary_writer=None):

        reward = None
        is_true_pred = None

        if session is None:
            pred_sns = self.sensor_agent.predict(self.cur_sns_input)
            pred_img = self.vision_agent.predict(self.cur_img_input)
        else:
            pred_sns = self.sensor_agent.predict_from_tf(self.cur_sns_input, session=session)
            pred_img = self.vision_agent.predict_from_tf(self.cur_img_input, session=session)

        if action == self.SENSOR:
            reward = self.calculate_reward(self.cur_sns_label, pred_sns, pred_img, self.SENSOR)
            self.current_consumption += self.sensor_consumption_per_step
            is_true_pred = True if pred_sns == np.argmax(self.cur_sns_label) else False

        if action == self.CAMERA:
            reward = self.calculate_reward(self.cur_img_label, pred_sns, pred_img, self.CAMERA)
            self.current_consumption += self.vision_consumption_per_step
            is_true_pred = True if pred_img == np.argmax(self.cur_img_label) else False

        # if verbose:
        #     print('Current Consumption: %f. - Action: %d' % (self.current_consumption, action))

        #done = self.current_consumption >= self.battery_size
        #state = self.sensor_agent.get_state_for_input(self.cur_sns_input)

        if action_probs is not None:
            self.ep_probs.append(action_probs)

        self.ep_steps.append(action)
        self.ep_rewards.append(reward)
        self.ep_true_preds.append(is_true_pred)

        state = self.cur_sns_input
        self.done, self.cur_img_input, self.cur_img_label, self.cur_sns_input, self.cur_sns_label = next(self.state_generator)

        if self.done:
            score = np.sum(self.ep_rewards, axis=0)

            self.all_scores.append(score)

            sensor_steps = np.where(np.array(self.ep_steps) == 0)[0]
            vision_steps = np.where(np.array(self.ep_steps) == 1)[0]

            self.all_rewards.append([np.array(self.ep_rewards)[sensor_steps].sum(),
                                np.array(self.ep_rewards)[vision_steps].sum()])

            self.all_true_preds.append([np.array(self.ep_true_preds).sum(), len(self.ep_true_preds)])

            score_mean = np.average(self.all_scores)
            self.all_action_avg.append(np.average(self.ep_probs, axis=0))
            action_avg = np.average(self.all_action_avg, axis=0)

            acc = np.array(self.ep_true_preds).sum() / float(len(self.ep_true_preds))
            self.all_acc.append(acc)
            acc_mean = np.average(self.all_acc)

            self.moving_average.append([score_mean, acc_mean])
            if self.logger is not None:
                self.logger.info('Episode: %d - Reward: %.2f - Avg Score: %.2f - Avg Accuracy: %.2f'
                            % (episode, score, score_mean, acc_mean))

                self.logger.info('Number of steps - Sensor: %d, Vision: %d - Avg rewards - Sensor: %.2f, Vision %.2f'
                            % (len(sensor_steps), len(vision_steps),
                               np.mean(self.all_rewards, axis=0)[0], np.mean(self.all_rewards, axis=0)[1]))

                self.logger.info('Action probability average - Sensor: %.2f Vision %.2f' % (action_avg[0], action_avg[1]))

            if summary_writer is not None:
                summary = tf.Summary()
                summary.value.add(tag="accuracy_mean", simple_value=float(acc_mean))
                summary.value.add(tag="sensor_rewards", simple_value=float(np.mean(self.all_rewards, axis=0)[0]))
                summary.value.add(tag="camera_rewards", simple_value=float(np.mean(self.all_rewards, axis=0)[1]))
                summary.value.add(tag="sensor_usage", simple_value=float(action_avg[0]))
                summary.value.add(tag="camera_usage", simple_value=float(action_avg[1]))
                summary_writer.add_summary(summary, episode)
                summary_writer.flush()

            self.ep_rewards = []
            self.ep_steps = []
            self.ep_probs = []
            self.ep_true_preds = []

            if episode > 1 and episode % 5 == 0:
                path = os.path.join('training_stats', 'alpha_'+str(self.alpha), self.current_time)
                os.makedirs(path, exist_ok=True)
                with h5py.File(os.path.join(path, self.env_id+'_stats_'+self.current_time+'.hdf5'), "w") as hf:
                    hf.create_dataset("scores", data=self.all_scores)
                    hf.create_dataset("moving_average", data=self.moving_average)
                    hf.create_dataset('batch_acc', data=self.all_acc)
                    hf.create_dataset("action_avg", data=self.all_action_avg)
                    hf.create_dataset("rewards", data=np.array(self.all_rewards))
                    hf.create_dataset("true_preds", data=self.all_true_preds)

        return state, reward, self.done

    def sample_from_episode(self):
        while True:
            img_x, img_y, sns_x, sns_y = next(self.episode_generator)

            img_x, img_y = MultimodalDataset.split_windows(self.img_chunk_size, self.img_chunk_size,
                                                           img_x[np.newaxis], img_y)

            sns_x, sns_y = MultimodalDataset.split_windows(self.sns_chunk_size, self.sns_chunk_size,
                                                           sns_x[np.newaxis], sns_y)
            for cur_ix, (i_x, i_y, s_x, s_y) in enumerate(zip(img_x, img_y, sns_x, sns_y)):
                done = True if cur_ix + 1 == len(img_x) else False
                yield done, i_x, i_y, s_x, s_y

    def _episode_generator(self):
        self.datasets_full_sweeps = 0
        while True:
            all_dirs = glob(os.path.join(self.img_root, '*', '*'))
            np.random.shuffle(all_dirs)

            for act_seq in all_dirs:
                img_x = []
                source_split_arr = act_seq.split(os.path.sep)
                sequence = source_split_arr[-1]
                activity = source_split_arr[-2]
                activity_number = activity_dict()[activity][0]
                sns_file = os.path.join(self.sns_root, activity+sequence+".csv")
                sns_x = np.squeeze(MultimodalDataset.load_sensor_from_file(sns_file, self.sensors))[:self.sns_max_samples]

                all_img = glob(os.path.join(act_seq, '*.jpg'))

                for ix, img_file in enumerate(all_img):
                    if ix < self.img_max_samples:
                        img_x.append(MultimodalDataset.get_img_from_file(img_file))

                img_y = np.eye(20)[np.repeat(activity_number, len(img_x))]
                sns_y = np.eye(20)[np.repeat(activity_number, len(sns_x))]

                yield np.array(img_x), img_y, sns_x, sns_y

            self.datasets_full_sweeps += 1

if __name__ == "__main__":
    act_env = ActivityEnvironment()
    act_env.reset()
    while True:
        act_env.step(act_env.SENSOR)
