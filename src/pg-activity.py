import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, LSTM
from keras.optimizers import Adam
from ActivityEnvironment import ActivityEnvironment
import h5py

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
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        #model.add(Reshape((1, self.state_size, 9), input_shape=(self.state_size, 9)))
        # model.add(Convolution2D(32, 6, 6, subsample=(3, 3), border_mode='same',
        #                         activation='relu', init='he_uniform'))
        #model.add(Flatten())
        model.add(LSTM(64, input_shape=(self.state_size, 9)))
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

    def act(self, state, stochastic=True):
        state = state[np.newaxis, :, :]
        aprob = self.model.predict(state, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        if stochastic is True:
            action = np.random.choice(self.action_size, 1, p=prob)[0]
        else:
            action = aprob.argmax()
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        #standardize the rewards to be unit normal (helps control the gradient estimator variance
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        rewards -= float(rewards_mean)
        rewards /= rewards_std if rewards_std != 0 else 1
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

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
            if action == env.CAMERA:
                pred = env.vision_agent.predict(img_x)
            preds.append(pred)
            true_y.append(y)
            true_preds = np.array(preds) == np.array(true_y)

    return ((true_preds.sum()/len(true_y))*100)

def train_policy():
    env = ActivityEnvironment(dataset_file='multimodal_full_train.hdf5',
                              sensor_model_weights='sensor_model.hdf5',
                              vision_model_weights='checkpoints/inception.029-1.08.hdf5',
                              split=False)
    load_weights = False
    all_scores = []
    score = 0
    episode = 0
    state = env.reset()

    state_size = 5
    action_size = 2

    agent = PGAgent(state_size, action_size)
    if load_weights:
        agent.model.load_weights('activity.h5')
        all_scores = h5py.File('scores.hdf5')['scores'][:].tolist()
        print('Last Average score: %f.' % (sum(all_scores) / float(len(all_scores))))
    best_acc = 0
    while True:
        action, prob = agent.act(state)
        state, reward, done = env.step(action, verbose=False)
        score += reward
        agent.remember(state, action, prob, reward)

        if done:
            episode += 1
            print('Action probability average: ', (np.average(agent.probs, axis=0)))
            agent.train()
            all_scores.append(score)
            acc = evaluate_policy()
            print("Current accuracy: ", (acc))
            print('Episode: %d - Reward: %f. - Avg Score: %f.' % (episode, score, sum(all_scores)/float(len(all_scores))))
            score = 0
            state = env.reset()
            if episode > 1 and episode % 20 == 0:
                if acc > best_acc:
                    agent.save('activity.h5')
                    best_acc = acc
                with h5py.File('scores.hdf5', "w") as hf:
                    hf.create_dataset("scores", data=all_scores)




if __name__ == "__main__":
    train_policy()