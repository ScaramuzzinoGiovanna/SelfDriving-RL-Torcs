import numpy as np


class Q_table():
    def __init__(self, speed_discr, angle_discr, dist_discr):
        self.dict_actions = {0: [-1, -1], 1: [-1, 0], 2: [-1, 1], 3: [0, -1], 4: [0, 0], 5: [0, 1], 6: [1, -1],
                             7: [1, 0], 8: [1, 1]}

        self.vect_states = np.array(np.meshgrid(speed_discr, angle_discr, dist_discr)).T.reshape(-1, 3)
        self.dict_states = {key: value.tolist() for key, value in enumerate(self.vect_states)}
        # self.q_table = self.inizialize_q_table()

    def inizialize_q_table(self, test=False, q_table=None, q_updates=None):
        if not test:
            size_states = len(self.dict_states.keys())
            size_actions = len(self.dict_actions.keys())
            self.q_table = np.zeros([size_states, size_actions])
            self.q_updates = np.zeros([size_states, size_actions])
        else:
            self.q_table = q_table
            self.q_updates = q_updates

    def get_q_table(self):
        return self.q_table

    def get_q_updates(self):
        return self.q_updates

    def set_q_value(self, state, action_value, new_value):
        action_key = self.get_key(action_value, self.dict_actions)
        state_key = self.get_key(state, self.dict_states)
        self.q_table[state_key, action_key] = new_value
        self.q_updates[state_key, action_key] += 1

    def get_q_value(self, state, action_value):
        action_key = self.get_key(action_value, self.dict_actions)
        state_key = self.get_key(state, self.dict_states)
        q_value = self.q_table[state_key, action_key]
        return q_value

    def get_max_action(self, state):
        action_key = np.argmax(self.q_table[self.get_key(state, self.dict_states), :])
        action_value = self.dict_actions[action_key]
        return action_value

    def get_max_q_value(self, state):
        q_value = np.max(self.q_table[self.get_key(state, self.dict_states), :])
        return q_value

    def get_updates(self, state, action_value):
        action_key = self.get_key(action_value, self.dict_actions)
        state_key = self.get_key(state, self.dict_states)
        q_up = self.q_updates[state_key, action_key]
        return q_up

    def get_key(self, val, d):
        for key, value in d.items():
            if np.all(val == value):
                return key
        return "key doesn't exist"
