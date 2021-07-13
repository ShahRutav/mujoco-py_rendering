import gym
from abc import ABC
import numpy as np

class PixelWrapper(gym.Env, ABC):
    def __init__(self, env, cameras, height=100, width=100, channels_first=False, device_id=0):
        self._env = env
        self.cameras = cameras
        self.height = height
        self.width = width
        self.channels_first = channels_first
        self.action_space = self._env.action_space
        self.device_id = device_id
        self.env_kwargs = {'cameras' : cameras, 'height' : height, 'width':width, 'channels_first':channels_first}

        shape = [len(cameras), 3, height, width] if channels_first else [len(cameras), height, width, 3]
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self.sim = env.sim
        self.horizon = env.spec.max_episode_steps
    
    def get_obs(self, state):
        #return state
        imgs = np.zeros(self._observation_space.shape, dtype=np.uint8)
        for ind, cam in enumerate(self.cameras) : 
            img = self.sim.render(width=self.width, height=self.height, mode='offscreen', camera_name=cam, device_id=self.device_id) 
            img = img[::-1, :, :]
            if self.channels_first :
                img = img.transpose((2, 0, 1))
            imgs[ind, :, :, :] = img 
        return imgs

    def get_env_infos(self):
        return self._env.get_env_infos()
    def set_seed(self, seed):
        return self._env.seed(seed)

    def reset(self):
        obs = self._env.reset()
        obs = self.get_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, env_info = self._env.step(action)
        obs = self.get_obs(obs)
        return obs, reward, done, env_info

