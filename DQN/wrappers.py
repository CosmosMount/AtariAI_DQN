import numpy as np
from collections import deque
import gymnasium as gym
import cv2
import ale_py

gym.register_envs(ale_py)
cv2.ocl.setUseOpenCL(False)

def make_atari_env(env_id,
                   episodic_life=True,
                   clip_rewards=True,
                   stack_frames=True,
                   scale=False):
    env = gym.make(
        env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode="rgb_array"
    )

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=episodic_life,
    )

    if stack_frames:
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    if clip_rewards:
        env = ClipRewardEnv(env)

    if scale:
        env = ScaledFloatFrame(env)

    return env

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)
    
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

    
