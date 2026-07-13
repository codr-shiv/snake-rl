import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0        # Exploration rate
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.994 # Decay rate suited for 1000 episodes
        self.gamma = 0.97         # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.tau = 0.01           # Target network soft update coefficient
        
        # Policy Network
        self.model = Linear_QNet(12, 256, 3)
        # Target Network (cloned from Policy Network)
        self.target_model = Linear_QNet(12, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

        # Polyak soft update: target_weights = tau * policy_weights + (1 - tau) * target_weights
        for target_param, policy_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, train=True):
        # Epsilon-greedy action selection
        # During play/evaluation, train is False, so we always exploit
        if train and (random.random() < self.epsilon):
            action = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():
                prediction = self.model(state_tensor)
            action = torch.argmax(prediction).item()
            
        return action

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        file_path = os.path.join('./model', file_name)
        if os.path.exists(file_path):
            self.model.load_state_dict(torch.load(file_path))
            self.model.eval()
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            print(f"Successfully loaded model from {file_path}")
        else:
            print(f"No model found at {file_path}")
