import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, LSTM
from keras.optimizers import Adam
from ActivityEnvironment import ActivityEnvironment
from SensorAgent import SensorAgent
from VisionAgent import VisionAgent
import h5py
import logging
import datetime
import sys
import argparse
class LinearDecayEpsilonGreedy():
    """Epsilon-greedy with linearyly decayed epsilon
    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, start_epsilon, end_epsilon,
                 decay_steps):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.epsilon = start_epsilon

    def select_action_epsilon_greedily(self, epsilon, probs):
        random_number = np.random.rand()
        if random_number < epsilon:
            return self.random_action_func()
        else:
            return self.greedy_action_func(probs)

    def greedy_action_func(self, probs):
        return np.argmax(probs)

    def random_action_func(self):
        return np.random.choice([0, 1], p=[0.5, 0.5])

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, probs):
        self.epsilon = self.compute_epsilon(t)
        a = self.select_action_epsilon_greedily(self.epsilon, probs)
        return a

    def __repr__(self):
        return 'LinearDecayEpsilonGreedy(epsilon={})'.format(self.epsilon)

class PGAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.steps = []
        self.true_preds = []
        self.model = self._build_model()
        self.model.summary()
        self.epsilon_greedy = LinearDecayEpsilonGreedy(1, 0.05, 100)


    def _build_model(self, batches=14):
        model = Sequential()
        #model.add(Reshape((1, self.state_size, 9), input_shape=(self.state_size, 9)))
        # model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
        #                         activation='relu', init='he_uniform'))
        #model.add(Flatten())
        model.add(LSTM(64, batch_input_shape=(1, self.state_size, 6), stateful=True))
        #model.add(Dense(64, activation='relu', init='he_uniform',input_shape=(self.state_size, )))
        #model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state, stochastic=True, t=0):
        state = state[np.newaxis, :, :]
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        if stochastic is True:
            action = self.epsilon_greedy.select_action(t, prob)
            #action = np.random.choice(self.action_size, 1, p=prob)[0]
            # action = np.argmax(prob)
            # epsilon_greedy = [0.9 if ix == action else 0.1 for ix in range(0, 2)]
            # action = np.random.choice(self.action_size, 1, p=epsilon_greedy )[0]
        else:
            action = aprob.argmax()
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            # if rewards[t] != 0:
            #     running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards).astype(np.float)
        #rewards = self.discount_rewards(rewards)
        #standardize the rewards to be unit normal (helps control the gradient estimator variance
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        rewards -= rewards_mean
        rewards /= rewards_std if rewards_std != 0 else 1
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))

        for x, y in zip(X,Y):

            if len(x.shape) < 3:
                x = x[np.newaxis, :, :]
            if len(y.shape) < 2:
                y = y[np.newaxis, :]

            self.model.train_on_batch(x, y)

        self.model.reset_states()
        self.states, self.probs, self.gradients, self.rewards,self.steps, self.true_preds = [], [], [], [], [], []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def evaluate_policy(dataset_file="multimodal_full_test.hdf5", agent_weights='activity.h5', state_size=5, action_size=2):
    env = ActivityEnvironment(dataset_file=dataset_file,
                              sensor_model_weights='sensor_model.hdf5',
                              vision_model_weights='checkpoints/inception.029-1.08.hdf5',
                              split=False)
    agent = PGAgent(state_size, action_size)
    if agent_weights is not None:
        agent.model.load_weights(agent_weights)
    preds = []
    true_y = []
    steps = [0, 0]
    for i in range(0, env.total_size):
        env.read_sensors([i])
        while not env.current_x_activity_sns_buffer.empty():
            sns_x = env.current_x_activity_sns_buffer.get()
            img_x = env.current_x_activity_img_buffer.get()
            y = env.current_y_activity_sns_buffer.get()
            state = sns_x
            action, prob = agent.act(state, stochastic=False)

            if action == env.SENSOR:
                pred = env.sensor_agent.predict(sns_x)
                steps[0] += 1
            if action == env.CAMERA:
                pred = env.vision_agent.predict(img_x)
                steps[1] += 1

            preds.append(pred)
            true_y.append(y)
            true_preds = np.array(preds) == np.array(true_y)

    print("Steps: ", steps)
    return ((true_preds.sum()/len(true_y))*100)


def train_policy(alpha, num_episodes=2000):
    sensor_agent = SensorAgent(weights="models/sensor_model.hdf5")
    vision_agent = VisionAgent(weights="models/vision_model.hdf5")

    env = ActivityEnvironment(sensor_agent=sensor_agent, vision_agent=vision_agent, alpha=alpha)
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(str(alpha)+'_train_policy_'+current_time+'.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    load_weights = False
    all_scores = []
    all_rewards = []
    all_true_preds = []
    all_action_avg = []
    moving_average = []
    score_mean = 0
    acc_mean = 0
    all_acc = []
    score = 0
    episode = 0
    state = env.reset()

    state_size = 10
    action_size = 2

    agent = PGAgent(state_size, action_size)
    if load_weights:
        agent.model.load_weights('activity.h5')
        all_scores = h5py.File('scores.hdf5')['scores'][:].tolist()
        logger.info('Last Average score: %.2f' % (sum(all_scores) / float(len(all_scores))))
    while episode < num_episodes:
        action, prob = agent.act(state, t=episode)
        agent.steps.append(action)
        state, reward, done, is_true_pred = env.step(action, verbose=False)
        agent.true_preds.append(is_true_pred)
        score += reward
        agent.remember(state, action, prob, reward)

        if done:
            episode += 1

            all_scores.append(score)

            sensor_steps = np.where(np.array(agent.steps) == 0)[0]
            vision_steps = np.where(np.array(agent.steps) == 1)[0]

            all_rewards.append([np.array(agent.rewards)[sensor_steps].sum(),
                                np.array(agent.rewards)[vision_steps].sum()])
            all_true_preds.append([np.array(agent.true_preds).sum(), len(agent.true_preds)])

            score_mean = np.average(all_scores)
            all_action_avg.append(np.average(agent.probs, axis=0))
            action_avg = np.average(all_action_avg, axis=0)

            acc = np.array(agent.true_preds).sum() / float(len(agent.true_preds))
            all_acc.append(acc)
            acc_mean = np.average(all_acc)

            moving_average.append([score_mean, acc_mean])

            logger.info('Episode: %d - Reward: %.2f - Avg Score: %.2f - Avg Accuracy: %.2f'
                        % (episode, score, score_mean, acc_mean))
            logger.info('Number of steps - Sensor: %d, Vision: %d'
                        % (len(sensor_steps), len(vision_steps)))
            logger.info('Action probability average - Sensor: %.2f Vision %.2f' % (action_avg[0], action_avg[1]))

            agent.train()

            score = 0
            state = env.reset()
            if episode > 1 and episode % 20 == 0:

                # if episode == 20:
                #     acc = evaluate_policy(agent_weights=None)
                # else:
                #     acc = evaluate_policy()

                agent.save(str(alpha)+'_activity_'+current_time+'.h5')

                with h5py.File(str(alpha)+'_stats_'+current_time+'.hdf5', "w") as hf:
                    hf.create_dataset("scores", data=all_scores)
                    hf.create_dataset("moving_average", data=moving_average)
                    hf.create_dataset('batch_acc', data=all_acc)
                    hf.create_dataset("action_avg", data=all_action_avg)
                    hf.create_dataset("rewards", data=np.array(all_rewards))
                    hf.create_dataset("true_preds", data=all_true_preds)

    logger.removeHandler(handler)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--grid_search", action="store_true")
    a.add_argument("--train_policy", action="store_true")
    a.add_argument("--evaluate_policy", action="store_true")
    a.add_argument("--alpha", default=0, type=int)
    a.add_argument("--num_episodes", default=20, type=int)
    args = a.parse_args()

    if args.train_policy:
        print("Single Instance Training for alpha {}".format(args.alpha))
        train_policy(alpha=args.alpha, num_episodes=args.num_episodes)

    if args.grid_search:
        alphas = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for alpha in alphas:
            print("Grid Search Training for alpha {}".format(alpha))
            train_policy(alpha=alpha, num_episodes=args.num_episodes)
