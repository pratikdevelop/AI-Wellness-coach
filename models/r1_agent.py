import numpy as np
from collections import defaultdict
import pickle
import os

class RLAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, q_table_path='q_table.pkl'):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.q_table_path = q_table_path
        self.load_q_table()

    def discretize_state(self, state):
        activity_level = min(int(state['activity_level']), 3)
        stress_level = min(int(state['stress_level'] / 3), 3)
        sleep_hours = min(int(state['sleep_hours'] / 2), 4)
        mood = int(state['mood'])  # Expect mood_encoded (0-3)
        return (activity_level, stress_level, sleep_hours, mood)

    def get_action(self, state):
        state_tuple = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = self.q_table[state_tuple]
            return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        state_tuple = self.discretize_state(state)
        next_state_tuple = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        
        current_q = self.q_table[state_tuple][action_idx]
        next_max_q = np.max(self.q_table[next_state_tuple])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_tuple][action_idx] = new_q

    def save_q_table(self):
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), pickle.load(f))