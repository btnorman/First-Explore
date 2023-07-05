import jax
import brax
from brax import jumpy as jp
from brax.envs import env

import optax
import flax
import jax.numpy as jnp
import math
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
import numpy as np
import flax.linen as nn


class Bandit(env.Env):
    def __init__(self, key=None, n=10, deterministic=True, noise_scale=0.1):
        if key is None:
            key = jax.random.PRNGKey(0)
        self.n = n
        self.arm_means = jax.random.normal(key, (n,))
        self.deterministic = deterministic
        self.noise_scale = noise_scale
        
    def gen_arm_means(self, rng):
        return jax.random.normal(rng, (self.n,))
        
    def reset(self, rng: jp.ndarray, arm_means=None) -> env.State:
        """Resets the environment to an initial state."""
        if arm_means is None:
            arm_means = self.arm_means
        return env.State(qp=rng, # ha, overloading this :p
                         obs=jp.array(()),
                         reward=jp.zeros((1,)),
                         done=jp.zeros((),dtype=bool),
                         metrics={},
                         info={'arm_means' : arm_means})
    
    def meta_reset(self, rng: jp.ndarray) -> env.State:
        """Meta resets the environment"""
        meta_key, reset_key = jax.random.split(rng, 2)
        arm_means = self.gen_arm_means(meta_key)
        return env.State(qp=reset_key,
                         obs=jp.array(()),
                         reward=jp.zeros((1,)),
                         done=jp.zeros((),dtype=bool),
                         metrics={},
                         info={'arm_means' : arm_means})

    def step(self, state : env.State, action: jp.ndarray) -> env.State:
        if self.deterministic:
            return state.replace(reward=state.info['arm_means'][action].reshape((1,)),
                                 done=True)
        else:
            key, s_key = jax.random.split(state.qp)
            reward = (state.info['arm_means'][action]
                      + self.noise_scale * jax.random.normal(s_key, (1,))).reshape((1,))
            return state.replace(reward=reward,
                                 done=True,
                                 qp=key)