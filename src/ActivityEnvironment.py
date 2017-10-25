import numpy as np
from MultimodalDataset import MultimodalDataset
from globals import activity_dict
import queue as q
from glob import glob
import os

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
                 alpha=0):

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

    def reset(self):
        self.current_consumption = 0
        self.done = False
        self.done, self.cur_img_input, self.cur_img_label, self.cur_sns_input, self.cur_sns_label = next(self.state_generator)
        return self.cur_sns_input

    def calculate_reward(self, real, pred_sns, pred_img, sensor_type):

        real = np.argmax(real)

        if sensor_type == self.SENSOR:
            # When Sensor is Right
            if pred_sns == real and pred_img != real:
                total_reward = self.reward_right_pred

            if pred_sns == real and pred_img == real:
                total_reward = 1 - self.alpha

            # When Sensor is Wrong
            if pred_sns != real and pred_img == real:
                total_reward = self.reward_wrong_pred - (1 - self.alpha)

            if pred_sns != real and pred_img != real:
                total_reward = self.reward_right_pred - (1 - self.alpha)

        if sensor_type == self.CAMERA:
            # When Camera is Right
            if pred_img == real and pred_sns != real:
                total_reward = self.reward_right_pred

            if pred_img == real and pred_sns == real:
                total_reward = self.alpha

            # When Camera is Wrong
            if pred_img != real and pred_sns == real:
                total_reward = self.reward_wrong_pred - self.alpha

            if pred_sns != real and pred_img != real:
                total_reward = self.reward_wrong_pred - self.alpha

        return total_reward


    def step(self, action, verbose=True):

        reward = None
        is_true_pred = None

        pred_sns = self.sensor_agent.predict(self.cur_sns_input)
        pred_img = self.vision_agent.predict(self.cur_img_input)

        if action == self.SENSOR:
            reward = self.calculate_reward(self.cur_sns_label, pred_sns, pred_img, self.SENSOR)
            self.current_consumption += self.sensor_consumption_per_step
            is_true_pred = True if pred_sns == np.argmax(self.cur_sns_label) else False

        if action == self.CAMERA:
            reward = self.calculate_reward(self.cur_img_label, pred_sns, pred_img, self.CAMERA)
            self.current_consumption += self.vision_consumption_per_step
            is_true_pred = True if pred_img == np.argmax(self.cur_img_label) else False

        if verbose:
            print('Current Consumption: %f. - Action: %d' % (self.current_consumption, action))

        done = self.current_consumption >= self.battery_size

        #state = self.sensor_agent.get_state_for_input(self.cur_sns_input)
        state = self.cur_sns_input
        self.done, self.cur_img_input, self.cur_img_label, self.cur_sns_input, self.cur_sns_label = next(self.state_generator)
        return state, reward, self.done, is_true_pred

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

if __name__ == "__main__":
    act_env = ActivityEnvironment()
    act_env.reset()
    while True:
        act_env.step(act_env.SENSOR)
