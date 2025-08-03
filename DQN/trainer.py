import os
import time
import csv
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym
from params import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm

from model import DQN
from wrappers import make_atari_env
from replay import ReplayBuffer
from params import *

device = torch.device("cuda")


class ModelTrainer:
    def __init__(self, env_names=['Boxing', 'Pong']):
        self.env_names = env_names
        self.device = device

        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Create log writers for each environment
        self.log_writers = {}
        for env_name in env_names:
            log_path = os.path.join("logs", f"{env_name}_log.csv")
            log_file = open(log_path, mode='w', newline='')
            writer = csv.writer(log_file)
            writer.writerow(['Step', 'Reward', 'Loss'])
            self.log_writers[env_name] = {'file': log_file, 'writer': writer}

    def __del__(self):
        for env_log in self.log_writers.values():
            env_log['file'].close()

    def compute_loss(self, batch_size, gamma):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state)).to(self.device)
        next_state = torch.FloatTensor(np.float32(next_state)).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values_old = self.model(state)
        q_values_new = self.model(next_state)

        q_value_old = q_values_old.gather(1, action.unsqueeze(1)).squeeze(1)
        q_value_new = q_values_new.max(1)[0]
        expected_q_value = reward + gamma * q_value_new * (1 - done)

        loss = (q_value_old - expected_q_value.data).pow(2).mean()
        return loss

    def train(self, env_name):
        env = make_atari_env(f"ALE/{env_name}-v5")
        self.model = DQN(env.observation_space.shape, env.action_space.n).to(device)    
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        steps_done = 0
        episode_rewards = []
        losses = []
        self.model.train()

        writer = self.log_writers[env_name]['writer']

        for episode in tqdm(range(1, EPISODES + 1), desc=f"Training {env_name}"):
            state, _ = env.reset()
            episode_reward = 0.0

            while True:
                epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-steps_done / EPS_DECAY)
                action = self.model.act(state, epsilon, self.device)
                steps_done += 1

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                loss_value = None
                if len(self.replay_buffer) > INITIAL_MEMORY:
                    loss = self.compute_loss(BATCH_SIZE, GAMMA)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    loss_value = loss.item()
                    losses.append(loss_value)

                writer.writerow([steps_done, reward, loss_value if loss_value is not None else ''])

                if done:
                    episode_rewards.append(episode_reward)
                    break

            if (episode + 1) % 4000 == 0 or episode == 2:
                path = os.path.join(MODEL_SAVE_PATH, f"{env_name}/episode_{episode + 1}.pth")
                print(f"Saving weights at Episode {episode + 1} for {env_name}...")
                torch.save(self.model.state_dict(), path)

        env.close()

    def run(self):
        for env_name in self.env_names:
            print(f"\n========== Starting training on {env_name} ==========")
            self.train(env_name)

if __name__ == "__main__":
    clear_output(True)
    print("Starting training...")
    
    trainer = ModelTrainer(env_names=['Breakout'])
    trainer.run()

    print("Training completed.")

