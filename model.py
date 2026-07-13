import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.model = model
        self.target_model = target_model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            # Add batch dimension if it is a single sample
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predict Q values for current states
        pred = self.model(state)

        # Clone predictions to form targets
        target = pred.clone()

        # Update target values using Bellman Equation
        # Using the target network for future Q values to prevent inflation loops
        with torch.no_grad():
            next_pred = self.target_model(next_state)

        for idx in range(len(done)):
            Q_new = reward[idx].item()
            if not done[idx]:
                Q_new = reward[idx].item() + self.gamma * torch.max(next_pred[idx]).item()

            # Update target value for the specific action chosen
            target[idx][action[idx]] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()
