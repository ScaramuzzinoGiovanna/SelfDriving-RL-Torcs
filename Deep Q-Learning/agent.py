from net import Net
from replay_buffer import ReplayBuffer
from gym import spaces
import random
import numpy as np
from modifiedTensorboard import ModifiedTensorBoard


def random_action():
    action = spaces.MultiDiscrete([3, 3]).sample()
    if action[0] == 2:
        action[0] = -1
    if action[1] == 2:
        action[1] = -1
    return action


def get_key(val, d):
    for key, value in d.items():
        if np.all(val == value):
            return key
    return "key doesn't exist"


class Agent():
    def __init__(self, test=False, state_size=3, action_size=9, batch_size=32, buffer_size=5000, min_buffer_size=1000,
                 learning_rate=0.001, discount=0.95, model_name=None):

        self.dict_actions = {0: [-1, -1], 1: [-1, 0], 2: [-1, 1], 3: [0, -1], 4: [0, 0], 5: [0, 1], 6: [1, -1],
                             7: [1, 0], 8: [1, 1]}
        if test == False:
            self.state_size = state_size
            self.action_size = action_size
            self.batch_size = batch_size
            self.discount = discount
            self.learning_rate_init = learning_rate
            self.learning_rate_decay = learning_rate

            self.buffer = ReplayBuffer(buffer_size, min_buffer_size)

            self.n_update = 10000
            self.target_update_counter = 0
            self.tensorboard = ModifiedTensorBoard(log_dir="logdir/{}".format(model_name))

    def initialize(self, test=False, run_intermedia=False, buff=None, n_exp=0, local=None, target=None):
        if test == True:
            self.qnet_local = local
        elif run_intermedia == False:
            self.buffer.initialize_buffer()
            self.qnet_local = Net(self.state_size, self.action_size, self.learning_rate_init).create_model()
            self.qnet_target = Net(self.state_size, self.action_size, self.learning_rate_init).create_model()
            self.qnet_target.set_weights(self.qnet_local.get_weights())
        else:
            self.buffer.initialize_buffer(buff, n_exp)
            self.qnet_local = local
            self.qnet_target = target

    def step(self, state, action, reward, next_state, done):

        action_key = get_key(action, self.dict_actions)
        self.buffer.add(state, action_key, reward, next_state, done)
        batch = self.buffer.getBatch(self.batch_size)
        if batch != 'small buffer':
            self.target_update_counter += 1
            if self.target_update_counter % 4 == 0 or done:
                self.train(done, batch)
        else:
            return

    def act(self, state, epsilon=0, test=False):
        action_values = self.qnet_local.predict_on_batch(x=state.reshape(1, state.shape[0]))  # x,batch_1

        if test == True:
            action = self.dict_actions[np.argmax(action_values)]
            return action

        elif random.uniform(0, 1) < epsilon:
            action = random_action()
        else:
            action = self.dict_actions[np.argmax(action_values)]
        return action

    def train(self, terminal_state, batch):
        states = np.asarray([e[0] for e in batch])
        next_states = np.asarray([e[3] for e in batch])
        current_qs_list = np.array(self.qnet_local.predict_on_batch(states))
        future_qs_list = np.array(self.qnet_target.predict_on_batch(next_states))
        X = np.zeros((self.batch_size, self.state_size))
        y = np.zeros((self.batch_size, self.action_size))

        for index, (state, action, reward, next_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X[index] = state
            y[index] = current_qs

        logs = self.qnet_local.train_on_batch(X, y)

        if terminal_state == True:
            map_logs = {'loss': logs[0], 'accuracy': logs[1]}
            self.tensorboard.update_stats(learning_rate=self.learning_rate_decay)
            self.tensorboard.on_epoch_end(self.tensorboard.step, map_logs)  # epoca #logs
        else:
            pass
        if terminal_state or self.target_update_counter >= 100:
            self.qnet_target.set_weights(self.qnet_local.get_weights())
            self.target_update_counter = 0

    def get_buffer(self):
        return self.buffer.get_buffer_and_exp()
