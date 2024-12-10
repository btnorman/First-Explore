import gym
from gym import spaces
## need to specify DiscreteAction Space
import numpy as np
import random


class TreasureEnv(gym.Env):
    def __init__(self, minval):
        super(TreasureEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=8, shape=(2,))
        self._minval = minval
        self._max_ep_length = 9
        self._max_episode_steps = 9
        self.reset_task()
        self.reset()
        self.task_dim = 8*3
        self.num_states = 9 ** 2
        
    def __gen_obs(self):
        return np.array((self._ax, self._ay))

    def reset(self):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """
        self._ep_steps_so_far = 0
        self._ax = 4
        self._ay = 4
        self._rds = np.zeros(8, dtype=bool)
        return self.__gen_obs()
    
    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        if task is None:
            self._rrs = np.array([random.uniform(self._minval, 2) for i in range(8)])
            self._rxs = np.array([random.randint(0, 8) for i in range(8)])
            self._rys = np.array([random.randint(0, 8) for i in range(8)])
        else:
            self._rrs = np.array(task[0:8])
            self._rxs = np.array(task[8:16], dtype=int)
            self._rys = np.array(task[16:], dtype=int)
        return self.get_task()

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return np.stack((self._rrs, self._rxs, self._rys)).reshape(-1)

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        self._ep_steps_so_far += 1
        a_t = action
        if a_t == 1:
            self._ay += 1
        elif a_t == 3:
            self._ay -= 1     
        elif a_t == 2:
            self._ax += 1
        elif a_t == 4:
            self._ax -= 1
        self._ax = np.clip(self._ax, 0, 8)
        self._ay = np.clip(self._ay, 0, 8)
        
        # check if we are the location of any unconsumed rewards
        rewards_got = (self._ax == self._rxs) * (self._ay == self._rys) * (~self._rds)
        reward = (self._rrs * rewards_got).sum() # give according reward
        self._rds = self._rds | rewards_got # update the list of consumsed rewards
        
        done_t = False if self._ep_steps_so_far < self._max_ep_length else True
#         if done_t and auto_reset:
#             self.reset()
        return self.__gen_obs(), reward, done_t, {'task' : self.get_task()}

    def task_to_id(self, obs):
        print(obs)
        return int(obs[0])*9+int(obs[1])
