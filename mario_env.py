import gym
import numpy as np
import retro
from gym import spaces
import cv2


class MarioEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.env = retro.make(game="SuperMarioBros-Nes", state="Level1-1")

        # Observation: (4, 84, 84)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, 84, 84),
            dtype=np.uint8
        )

        self.action_space = self.env.action_space
        self.frame_stack = []

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self):
        obs = self.env.reset()
        processed = self.preprocess(obs)

        self.frame_stack = [processed for _ in range(4)]
        return np.array(self.frame_stack, dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        processed = self.preprocess(obs)

        self.frame_stack.pop(0)
        self.frame_stack.append(processed)

        stacked = np.array(self.frame_stack, dtype=np.uint8)

        # reward shaping (progress-based)
        reward += info.get("x_pos", 0) * 0.001

        return stacked, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()