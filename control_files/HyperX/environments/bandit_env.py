import gym
from gym import spaces
## need to specify DiscreteAction Space
import numpy as np
import random


from typing import Tuple

import numpy as np

import gym
from gym import spaces
## need to specify DiscreteAction Space
import numpy as np
import random


class BanditEnv(gym.Env):
    def __init__(self, minval=0):
        super(BanditEnv, self).__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=0, shape=(2,))
        self._max_ep_length = 1
        self._max_episode_steps = 1
        self.reset_task()
        self.reset()
        self.task_dim = 10
        self.num_states = 1
        
    def __gen_obs(self):
        return np.zeros((2,))

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        return self.__gen_obs()
    
    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        if task is None:
            self._means = np.array([random.gauss(0, 1) for i in range(10)])
        else:
            self._means = np.copy(task)
        return self.get_task()

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return np.copy(self._means)

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        reward = self._means[action] + 0.5*random.gauss(0, 1)
        return self.__gen_obs(), reward.reshape(()), True, {'task' : self.get_task(), 'means' : self._means, 'action' : action}

    
class MeanBanditEnv(gym.Env):
    def __init__(self, minval=0):
        super(MeanBanditEnv, self).__init__()
        self.minval = minval
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=0, shape=(2,))
        self._max_ep_length = 1
        self._max_episode_steps = 1
        self.reset_task()
        self.reset()
        self.task_dim = 10
        self.num_states = 1

        
    def __gen_obs(self):
        return np.zeros((2,))

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        return self.__gen_obs()
    
    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        if task is None:
            self._means = np.array([random.gauss(0, 1) for i in range(10)])
            self._means[0] = self.minval
        else:
            self._means = np.copy(task)
        return self.get_task()

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return np.copy(self._means)

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        if action == 0:
            reward = self._means[action] # is this sensible?
        else:
            reward = self._means[action] + 0.5*random.gauss(0, 1)
        return self.__gen_obs(), reward.reshape(()), True, {'task' : self.get_task(), 'means' : self._means, 'action' : action}
