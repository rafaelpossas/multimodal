import h5py
import numpy as np
from MultimodalDataset import MultimodalDataset
from SensorAgent import SensorAgent
from VisionAgent import VisionAgent
import random
import queue as q


class EgocentricEnvironment(object):

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

    def __init__(self, dataset_file=None, sensor_model_weights=None,vision_model_weights=None,
                 chunk_size=15, step_size=15, split=True):

        self.dataset_file = h5py.File(dataset_file)
        self.multimodal_dataset = MultimodalDataset()
        self.vision_agent = None

        self.sns_x = self.dataset_file['x_sns'][:]
        self.sns_y = self.dataset_file['y'][:]
        self.onehot_y = np.eye(20)[self.sns_y.astype("int")]
        self.total_size = len(self.sns_x)
        self.current_index = 0

        self.batch_size_sns = int(self.num_sns_per_activity / self.sns_x.shape[1])
        self.batch_size_img = int(self.num_img_per_activity / self.dataset_file['x_img'].shape[1])

        self.sns_total_samples = self.dataset_file['x_sns'].shape[1]
        self.img_total_samples = self.dataset_file['x_img'].shape[1]
        print("Batch Size per activity: "+str(self.batch_size_sns))

        max_samples = max([self.sns_total_samples, self.img_total_samples])
        min_samples = min([self.sns_total_samples, self.img_total_samples])

        min_max_raio = max_samples / min_samples

        self.chunk_size_sensor = int(chunk_size if self.sns_total_samples > self.img_total_samples else chunk_size / min_max_raio)
        self.chunk_size_image = int(chunk_size if self.img_total_samples > self.sns_total_samples else chunk_size / min_max_raio)

        self.step_size_sensor = int(step_size if self.sns_total_samples > self.img_total_samples else step_size / min_max_raio)
        self.step_size_image = int(step_size if self.img_total_samples > self.sns_total_samples else step_size / min_max_raio)

        self.sensor_consumption_per_step = (self.sensor_consumption_per_hour/3600) * \
                                           ((self.chunk_size_sensor*self.total_seconds)/self.dataset_file['x_sns'].shape[1])
        self.vision_consumption_per_step = (self.camera_consumption_per_hour/3600) * \
                                           ((self.chunk_size_image*self.total_seconds)/self.dataset_file['x_img'].shape[1])


        print("Initial Shape:")
        print(self.sns_x.shape)
        print(self.onehot_y.shape)

        if split and (chunk_size != self.sns_x.shape[1] or chunk_size != step_size):
            original_x = self.sns_x.reshape((int(self.sns_x.shape[0] / 30), self.sns_x.shape[1] * 30,
                                             self.sns_x.shape[2]))
            original_one_hot_y = np.array([y for ix, y in enumerate(self.onehot_y) if ix % 30 == 0])
            self.sns_x, self.onehot_y = self.multimodal_dataset.split_windows(chunk_size, step_size, original_x,
                                                                              original_one_hot_y)
        print("Transformed Shape:")
        print(self.sns_x.shape)
        print(self.onehot_y.shape)

        self.sensor_agent = SensorAgent(model_weights=sensor_model_weights,
                                        input_shape=(self.chunk_size_sensor, self.sns_x.shape[2]),
                                        output_shape=self.onehot_y.shape[1])

        self.vision_agent = VisionAgent(model_weights=vision_model_weights)

    def get_random_index(self, total_size, batch_size=1):
        index = random.sample(range(0, total_size), batch_size)
        return index

    def reset(self):
        self.current_consumption = 0
        self.done = False
        self.get_next_from_buffer()
        #return self.sensor_agent.get_state_for_input(self.cur_sns_input)
        return self.cur_sns_input

    def image_generator(self, index=None):
        while True:
            if index is None:
                index = self.get_random_index(self.dataset_file['x_img'].shape[0], self.batch_size_img)
            x = self.dataset_file['x_img'][sorted(index)]
            y = self.dataset_file['y_img'][sorted(index)]
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
            y = np.eye(20)[np.repeat(y, int(self.num_img_per_activity/self.batch_size_img))]
            x = x.astype("float") / 255.0
            yield x, y

    def sensor_generator(self, index=None):
        while True:
            if index is None:
                index = self.get_random_index(self.dataset_file['x_sns'].shape[0], self.batch_size_sns)
            x = self.dataset_file['x_sns'][sorted(index)]
            y = self.dataset_file['y_sns'][sorted(index)]
            yield x, y

    def calculate_reward(self, real, pred_sns, pred_img, sensor_type):

        if sensor_type == self.SENSOR:
            if pred_sns == real and pred_img != real:
                total_reward = self.reward_right_pred
            if pred_sns != real and pred_img == real:
                total_reward = self.reward_wrong_pred
            if pred_sns == real and pred_img == real:
                total_reward = 1

        if sensor_type == self.CAMERA:
            if pred_img == real and pred_sns != real:
                total_reward = self.reward_right_pred
            if pred_img != real and pred_sns == real:
                total_reward = self.reward_wrong_pred
            if pred_img == real and pred_sns == real:
                total_reward = 0.8

        if pred_sns != real and pred_img != real:
            total_reward = 0

        return total_reward

    def read_sensors(self, index=None):
        x_sns, y_sns = next(self.sensor_generator(index=index))
        x_sns, y_sns = self.multimodal_dataset.split_windows(self.chunk_size_sensor, self.step_size_sensor, x_sns,
                                                             y_sns)
        # x_sns = self.multimodal_dataset.greedy_split(x_sns.reshape((x_sns.shape[1], x_sns.shape[2])), 5, 5)

        [self.current_x_activity_sns_buffer.put(sns) for sns in x_sns]
        [self.current_y_activity_sns_buffer.put(lbl) for lbl in y_sns]

        x_img, y_img = next(self.image_generator(index=index))
        x_img = self.multimodal_dataset.greedy_split(x_img, self.chunk_size_image, self.step_size_image)
        y_img = np.repeat(np.argmax(y_img[0]), len(x_img))

        [self.current_x_activity_img_buffer.put(img) for img in x_img]
        [self.current_y_activity_img_buffer.put(lbl) for lbl in y_img]

    def get_next_from_buffer(self, index=None):

        if self.current_x_activity_sns_buffer.empty() or self.current_x_activity_img_buffer.empty():
            if index is None:
                #index = self.get_random_index(self.total_size, 1)
                index = [self.current_index]
                if self.current_index < self.total_size:
                    self.current_index += 1
                else:
                    self.current_index = 0

            self.read_sensors(index)

        self.cur_sns_input = self.current_x_activity_sns_buffer.get()
        self.cur_img_input = self.current_x_activity_img_buffer.get()
        self.cur_label = self.current_y_activity_sns_buffer.get()

    def step(self, action, verbose=True):

        #print("Current img Buffer Size: ", self.current_x_activity_img_buffer.qsize())
        #print("Current sns Buffer Size: ", self.current_x_activity_sns_buffer.qsize())
        pred_sns = self.sensor_agent.predict(self.cur_sns_input)
        pred_img = self.vision_agent.predict(self.cur_img_input)

        if action == self.SENSOR:
            reward = self.calculate_reward(self.cur_label, pred_sns, pred_img, self.SENSOR)
            self.current_consumption += self.sensor_consumption_per_step
            is_true_pred = True if pred_sns == self.cur_label else False

        if action == self.CAMERA:
            reward = self.calculate_reward(self.cur_label, pred_sns, pred_img, self.CAMERA)
            self.current_consumption += self.vision_consumption_per_step
            is_true_pred = True if pred_img == self.cur_label else False

        if verbose:
            print('Current Consumption: %f. - Action: %d' % (self.current_consumption, action))

        done = self.current_consumption >= self.battery_size
        self.get_next_from_buffer()
        #state = self.sensor_agent.get_state_for_input(self.cur_sns_input)
        state = self.cur_sns_input

        return state, reward, done, is_true_pred


if __name__=="__main__":
    env = EgocentricEnvironment(dataset_file='../data/multimodal_full_test.hdf5',
                              sensor_model_weights='models/sensor_model.hdf5',
                              vision_model_weights='models/vision_weights_and_model.hdf5',
                              split=False)