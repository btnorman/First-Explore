import jax.numpy as jnp
import jax
from flax import struct

############
# A Dark Room Environment
# the room is w by h
# the agent recieves its x and y coordinates as observations
# and is rewarded at one location only
# the location of reward varies based on each Dark Room
# each time the agent moves

# In algorithm distilation the setup is thus:
# Agent starts in middle of room.
# Room Size either: 9x9 (darkroom) or 17x17 (darkroom hard)
# Actions: Left, Right, Up, Down, No-Op
# Reward: +1 if at correct location, else 0 (darkroom)
#         +1 if at goal for first time, else 0 (darkroom hard)
# Episode length is 20


@struct.dataclass
class RoomState:
    ax : jnp.ndarray # agent x
    ay : jnp.ndarray # agent y
    rx : jnp.ndarray # reward x
    ry : jnp.ndarray # reward y
    n : jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done : jnp.ndarray



class BatchedDarkRoom:
    def __init__(self, key, w, h, batch_size=1, rand_start=False, hard_reward=False):
        self.rand_start = rand_start
        self.hard_reward = hard_reward
        self.w = w
        self.h = h
        xkey, ykey = jax.random.split(key, 2)
        self.batch_size = batch_size
        self.rx = jax.random.randint(minval=0, maxval=w, shape=(batch_size,), key=xkey)
        self.ry = jax.random.randint(minval=0, maxval=h, shape=(batch_size,), key=ykey)
    
    
    def get_obs(self, ax, ay):
        return jnp.stack((ax, ay), axis=1)
        
    def meta_reset(self, key):
        rxkey, rykey, axkey, aykey = jax.random.split(key, 4)
        rx = jax.random.randint(minval=0, maxval=self.w,
                                shape=(self.batch_size,), key=rxkey)
        ry = jax.random.randint(minval=0, maxval=self.h,
                                shape=(self.batch_size,), key=rykey)
        if self.rand_start:
            ax = jax.random.randint(minval=0, maxval=self.w,
                                    shape=(self.batch_size,), key=axkey)
            ay = jax.random.randint(minval=0, maxval=self.h,
                                    shape=(self.batch_size,), key=aykey)
        else:
            ax = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.w//2)
            ay = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.h//2)
        return RoomState(ax=ax, ay=ay, rx=rx, ry=ry,
                         n=0, obs=self.get_obs(ax, ay),
                         reward=jnp.zeros(self.batch_size),
                         done=jnp.zeros((), dtype=bool))
  
    def step(self, state : RoomState, action):
        # action in 0, 1, 2, 3, 4
        # 0 no-op
        # 1 up, 2 right, 3 down, 4 left
        ax = jnp.clip(state.ax + (action==2) - (action==4), a_min=0, a_max=self.w-1)
        ay = jnp.clip(state.ay + (action==1) - (action==3), a_min=0, a_max=self.h-1)
        
        ## do the hard version here
        if self.hard_reward:
            reward = state.reward + ((ax==state.rx) & (ay==state.ry)) * (state.reward == 0)
        else:
            reward = state.reward + ((ax==state.rx) & (ay==state.ry))
        n = state.n+1
        obs = self.get_obs(ax, ay)
        done = n==20
        return state.replace(ax=ax, ay=ay, n=n, obs=obs,
                             reward=reward, done=done)
    
    def reset(self, state : RoomState, key):
        if self.rand_start:
            ax = jax.random.randint(minval=0, maxval=w,
                                    shape=(self.batch_size,), key=axkey)
            ay = jax.random.randint(minval=0, maxval=w,
                                    shape=(self.batch_size,), key=aykey)
        else:
            ax = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.w//2)
            ay = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.h//2)
        return state.replace(ax=ax, ay=ay,
                             n=0, obs=self.get_obs(ax, ay),
                             reward=jnp.zeros(self.batch_size),
                             done=jnp.zeros((), dtype=bool))
    

    def visualize(self, state: RoomState, i):
        room = jnp.zeros((self.w, self.h))
        room = room.at[state.rx[i], state.ry[i]].set(2)
        room = room.at[state.ax[i], state.ay[i]].set(1)
        return room

    
    def stack_vis(self, state):
        room = jnp.zeros((self.w, self.h, 3))
        cd = 1/len(state.rx)
        
        def stack_up_a(room, xy):
            x, y = xy
            room = room.at[x, y, 1].set(room[x, y, 1] + cd)
            return room, None
        def stack_up_d(room, xy):
            x, y = xy
            room = room.at[x, y, 0].set(room[x, y, 0] + cd)
            return room, None
        
        room, _ = jax.lax.scan(stack_up_a, room, (state.ax, state.ay))
        room, _ = jax.lax.scan(stack_up_d, room, (state.rx, state.ry))
        return room


import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera


def save_rollout_gif(env, states, i, file_name):
    fig = plt.figure()
    camera = Camera(fig)
    for state in states:
        room = env.visualize(state, i)
        plt.imshow(np.array(room))
        camera.snap()
    animation = camera.animate()
    animation.save(f"{file_name}.gif", dpi=150, writer="imagemagick")

    
def save_rollout_multi_gif(env, states, file_name, factor=1):
    fig = plt.figure()
    camera = Camera(fig)
    for state in states:
        room = env.stack_vis(state)**factor
        plt.imshow(np.array(room))
        camera.snap()
    animation = camera.animate()
    animation.save(f"{file_name}.gif", dpi=150, writer="imagemagick")