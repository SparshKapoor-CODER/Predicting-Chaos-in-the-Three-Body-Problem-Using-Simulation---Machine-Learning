# ml_rl_classification_pytorch.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from collections import deque
import random
from pathlib import Path
import os


# Create directories if they don't exist
Path("preprocessors/").mkdir(parents=True, exist_ok=True)
Path(os.path.dirname("models/three_body_rl_agent_pytorch.pth")).mkdir(parents=True, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Three-Body Environment
class ThreeBodyEnv(gym.Env):
    def __init__(self, df, max_steps=10):
        super(ThreeBodyEnv, self).__init__()
        self.df = df
        self.max_steps = max_steps
        self.current_step = 0
        self.current_idx = 0
        
        # Feature and label processing
        self.features = df.drop('label', axis=1).values
        self.labels = df['label'].values
        
        # Normalize features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)
        
        # Action space: Predict outcome (0: collision, 1: stable, 2: escape)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 21 features (masses, positions, velocities)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                           shape=(self.features.shape[1],))
        
        # State tracking
        self.state = None
        self.true_label = None
        
    def reset(self):
        self.current_step = 0
        self.current_idx = np.random.randint(0, len(self.df))
        self.state = self.features[self.current_idx]
        self.true_label = self.encoded_labels[self.current_idx]
        return self.state
    
    def step(self, action):
        reward = 10 if action == self.true_label else -1
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        self.current_idx = np.random.randint(0, len(self.df))
        self.state = self.features[self.current_idx]
        self.true_label = self.encoded_labels[self.current_idx]
        
        return self.state, reward, done, {}

# Deep Q-Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.tau = 0.01  # For soft update
        
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.update_target_net()
        
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main training function
def train_rl_classifier(df, episodes=1000, batch_size=64):
    env = ThreeBodyEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    rewards = []
    accuracies = []
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        correct_predictions = 0
        total_predictions = 0
        
        for time in range(env.max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            total_reward += reward
            if reward > 0:
                correct_predictions += 1
            total_predictions += 1
            
            state = next_state
            
            if done:
                break
        
        episode_accuracy = correct_predictions / total_predictions
        rewards.append(total_reward)
        accuracies.append(episode_accuracy)
        
        print(f"Episode: {e+1}/{episodes}, Reward: {total_reward:.2f}, "
              f"Accuracy: {episode_accuracy:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Plot training results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracy per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    plt.close()


    np.savez('preprocessors/scaler_params.npz', 
            mean=env.scaler.mean_, 
            scale=env.scaler.scale_)
    np.save('preprocessors/label_encoder_classes.npy', 
            env.label_encoder.classes_)
    
    return agent, env

# Evaluation function
def evaluate_agent(agent, env, test_size=0.2):
    test_indices = np.random.choice(len(env.df), int(len(env.df) * test_size), replace=False)
    test_features = env.features[test_indices]
    test_labels = env.encoded_labels[test_indices]
    
    predictions = []
    for feature in test_features:
        action = agent.act(feature)
        predictions.append(action)
    
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(test_labels, predictions)
    class_names = env.label_encoder.classes_
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.close()
    
    return accuracy, cm



# Feature importance analysis
def analyze_feature_importance(agent, env):
    baseline = np.mean(env.features, axis=0)
    importances = []
    
    for i in range(env.observation_space.shape[0]):
        test_state = baseline.copy()
        perturbation = np.std(env.features[:, i])
        test_state[i] += perturbation
        
        base_pred = agent.act(baseline)
        perturb_pred = agent.act(test_state)
        
        importances.append(1 if base_pred != perturb_pred else 0)
    
    feature_names = env.df.drop('label', axis=1).columns.tolist()
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title('Feature Importance based on RL Agent Sensitivity')
    plt.xlabel('Importance (1 = prediction changed)')
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return importances, feature_names

    

# Main execution
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv('three_body_dataset.csv')
    
    # Train RL agent
    agent, env = train_rl_classifier(df)
    
    # Evaluate performance
    accuracy, cm = evaluate_agent(agent, env)
    
    # Analyze feature importance
    importances, feature_names = analyze_feature_importance(agent, env)
    
    # Save model
    torch.save(agent.policy_net.state_dict(), 'models/three_body_rl_agent_pytorch.pth')
    print("RL agent trained and saved successfully!")

    