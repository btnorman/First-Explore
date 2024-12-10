from environments.maze3 import TinyWorld
import jax.random as jandom
import jax.numpy as jnp
import jax

import gym
from gym import spaces
import numpy as np
import random

#task:
# - gx_min, 3
# - gx_max, 3
# - gy_min, 3
# - gy_max, 3
# - hedges, 3x4 #(n.b. spare dims)
# - vedges, 4x3 #(n.b. spare dims)
# - goal_good, 3

class TinyWorldEnv(gym.Env):
    def __init__(self, side_rays=7, p_reward=0.3, max_ep_length=20, key=None, penalty_value=-1):
        super(TinyWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=5, shape=(2*(side_rays*2+1),))
        self.env = TinyWorld(side_rays=side_rays, p_reward=p_reward, batch_size=1, penalty_value=penalty_value)
        self.env_step = jax.jit(self.env.step)
        self.env_reset = jax.jit(self.env.reset)
        self.env_mreset = jax.jit(self.env.meta_reset)
        self._max_ep_length = max_ep_length
        self._max_episode_steps = max_ep_length
        
        
        if key is None:
            self.key = jandom.PRNGKey(random.randint(0, 2**63))
        else:
            self.key = key
        self.state = self.env.meta_reset(self.key)
        self.task_dim = 3*4+3*4+4*3+3
        self.num_states = 9*4*8 # estimate
        
    def reset(self):
        self.state = self.env_reset(self.state)
        return np.asarray(self.state.obs[0])
    
    def get_task(self):
        s = self.state
        return np.asarray(jnp.concatenate((s.gx_min, s.gx_max, s.gy_min, s.gy_max,
                                s.hedges, s.vedges, s.goal_good), axis=None))
    
    def reset_task(self, task=None):
        """task is an 3*4+3*4+4*3+3 dim array
        n.b. batchsize 1 environment under the hood"""

        state = self.state
        if task is None:
            self.key, m_key = jandom.split(self.key)
            self.state = self.env_mreset(m_key)
        else:
            self.state = state.replace(gx_min = task[None, 0:3],
                                       gx_max = task[None, 3:6],
                                       gy_min = task[None, 6:9],
                                       gy_max = task[None, 9:12],
                                       hedges = task[12:24].reshape((1, 3, 4)).astype(int),
                                       vedges = task[24:36].reshape((1, 4, 3)).astype(int),
                                       goal_good = task[None, 36:].astype(bool))
        return self.get_task()
    
    def step(self, action):
        self.state = self.env_step(self.state, jnp.array([[action,],])) 
        done_t = False if self.state.n < self._max_ep_length else True
        return np.asarray(self.state.obs[0]), np.asarray(self.state.reward[0])[0], done_t, {'task' : self.get_task()}