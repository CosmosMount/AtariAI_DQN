import os
import time
import torch
import torch.optim as optim

from IPython.display import clear_output
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import gymnasium as gym

from model import DQN
from wrappers import make_atari_env
from replay import ReplayBuffer
from params import *

device = torch.device("cuda")

class ModelPlayer:
    def __init__(self, env_name=['Boxing', 'Pong']):
        self.env_names = env_name
        self.device = device

        os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)

    def get_model_paths(self, env_name):
        model_dir = os.path.join(MODEL_SAVE_PATH, env_name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No model directory found for {env_name}: {model_dir}")
        return sorted([
            os.path.join(model_dir, fname)
            for fname in os.listdir(model_dir)
            if fname.endswith(".pth")
        ])

    def play(self, env_name):
        model_paths = self.get_model_paths(env_name)

        for model_path in model_paths:
            model_filename = os.path.splitext(os.path.basename(model_path))[0]  # e.g., Pong_episode_500
            video_dir = os.path.join(VIDEO_SAVE_PATH, env_name)
            os.makedirs(video_dir, exist_ok=True)

            # Setup environment with video recording
            env = make_atari_env(f"ALE/{env_name}-v5")
            env = RecordVideo(
                env,
                video_folder=video_dir,
                name_prefix=model_filename,
                episode_trigger=lambda e: True
            )

            # Load model
            model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()

            print(f"\n[Playing] {env_name} - Model: {model_filename}")

            for episode in range(5):
                state, _ = env.reset()
                episode_reward = 0.0

                while True:
                    action = model.act(state, epsilon=0.0, device=self.device)
                    next_state, reward, terminated, truncated, _ = env.step(action)

                    # Optional render
                    env.render()
                    time.sleep(0.02)

                    episode_reward += reward
                    state = next_state
                    if terminated or truncated:
                        print(f"Episode reward: {episode_reward}")
                        break

            env.close()

    def run(self):
        for name in self.env_names:
            print(f"\n========== Starting playback on {name} ==========")
            self.play(name)

if __name__ == "__main__":
    player = ModelPlayer(env_name=['Breakout'])
    player.run()
    print("Playback completed.")
        
