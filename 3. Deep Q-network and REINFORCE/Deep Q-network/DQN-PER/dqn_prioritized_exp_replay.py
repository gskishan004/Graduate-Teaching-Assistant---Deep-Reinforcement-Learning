import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


class dqn_prioritized_exp_replay:
    def __init__(
            self,
            actions=[(-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), (-1, 1, 0), (0, 1, 0), (1, 1, 0), (-1, 0, 0.2),
                     (0, 0, 0.2), (1, 0, 0.2), (-1, 0, 0), (0, 0, 0), (1, 0, 0)], stack=3, mem=5000, gamma=0.95,
            epsilon=1.0, epsilon_min=0.1, epsilon_decrease=0.99, alpha=0.001, prob = 0.6, initial_beta = 0.4
    ):
        self.action_space = actions
        self.frame_stack_num = stack
        self.memory = deque(maxlen=mem)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decrease
        self.learning_rate = alpha
        self.model = self.build()
        self.target_model = self.build()
        self.prob = prob
        self.initial_beta = initial_beta
        self.beta = initial_beta
        self.priorities = np.ones((mem,))
        self.update_target()

    def build(self):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu',
                         input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate, epsilon=1e-7))
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def memo(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            act_values = self.model.predict(np.expand_dims(state, axis=0))
            action_index = np.argmax(act_values[0])
        else:
            action_index = random.randrange(len(self.action_space))
        return self.action_space[action_index]

    def prioritized_replay_buffer(self, batch_size):
        minibatch = []
        indices = []
        priorities = self.priorities[:len(self.memory)] if len(self.memory) < len(self.priorities) else self.priorities
        for i in range(batch_size):
            probs = priorities ** self.prob
            probs /= probs.sum()
            index = np.random.choice(np.arange(len(self.memory)), p=probs)
            indices.append(index)
            minibatch.append(self.memory[index])
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        self.update_priorities(priorities, indices)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_priorities(self, priorities, indices):
        self.priorities[indices] = np.maximum(1e-5, priorities[indices])

    def update_beta(self, step):
        beta = self.initial_beta + step * (1.0 - self.initial_beta) / self.max_frames
        self.beta = min(1.0, beta)

    def load(self, name):
        self.model.load_weights(name)
        self.update_target()

    def save(self, name):
        self.target_model.save_weights(name)
