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
    rxs : jnp.ndarray # reward x
    rys : jnp.ndarray # reward y
    rrs : jnp.ndarray # reward r
    rds : jnp.ndarray # if reward has been gotten
    rgs : jnp.ndarray # if reward was gotten this step
    n : jnp.ndarray
    obs: jnp.ndarray
    reward: jnp.ndarray
    done : jnp.ndarray


@jax.vmap
def check(ax, ay, rxs, rys, rrs, rds):
    cond = ~rds & (rxs == ax) & (rys == ay)
    return jnp.where(cond, rrs, jnp.zeros_like(rrs)).sum(), cond


class BatchedDarkRoom:
    def __init__(self, w, h, batch_size=128, k=2, rand_start=False, reward_once=True, minval=-2, maxval=2):
        self.rand_start = rand_start
        self.w = w
        self.h = h
        self.k = k
        self.batch_size = batch_size
        self.reward_once=reward_once
        self.minval = minval
        self.maxval = maxval
    
    def get_obs(self, ax, ay):
        return jnp.stack((ax, ay), axis=1)
        
    def meta_reset(self, key):
        rxkey, rykey, rrkey, axkey, aykey = jax.random.split(key, 5)
        rx = jax.random.randint(minval=0, maxval=self.w,
                                shape=(self.batch_size, self.k),
                                key=rxkey)
        ry = jax.random.randint(minval=0, maxval=self.h,
                                shape=(self.batch_size, self.k),
                                key=rykey)
        rr = jax.random.uniform(minval=self.minval,
                                maxval=self.maxval,
                                shape=(self.batch_size, self.k),
                                key=rrkey)
        if self.rand_start:
            ax = jax.random.randint(minval=0, maxval=self.w,
                                    shape=(self.batch_size,), key=axkey)
            ay = jax.random.randint(minval=0, maxval=self.h,
                                    shape=(self.batch_size,), key=aykey)
        else:
            ax = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.w//2)
            ay = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.h//2)
        return RoomState(ax=ax, ay=ay,
                         rxs=rx, rys=ry, rrs=rr,
                         rds=jnp.zeros((self.batch_size, self.k,), dtype=bool),
                         rgs=jnp.zeros((self.batch_size, self.k,), dtype=bool),
                         n=0, obs=self.get_obs(ax, ay),
                         reward=jnp.zeros(self.batch_size),
                         done=jnp.zeros((), dtype=bool))
  
    def step(self, state : RoomState, action):
        # action in 0, 1, 2, 3, 4
        # 0 no-op
        # 1 up, 2 right, 3 down, 4 left
        ax = jnp.clip(state.ax + (action==2) - (action==4), a_min=0, a_max=self.w-1)
        ay = jnp.clip(state.ay + (action==1) - (action==3), a_min=0, a_max=self.h-1)
        
        reward_delta, rgs = check(ax=ax, ay=ay,
                                  rxs=state.rxs, rys=state.rys,
                                  rrs=state.rrs, rds=state.rds)
        reward = state.reward + reward_delta
        n = state.n+1
        obs = self.get_obs(ax, ay)
        done = n==20
        return state.replace(ax=ax, ay=ay, n=n, obs=obs,
                             rds=state.rds | rgs if self.reward_once else state.rds,
                             rgs=rgs,
                             reward=reward,
                             done=done)
    
    def reset(self, state : RoomState, key):
        if self.rand_start:
            axkey, aykey = jax.random.split(key, 2)
            ax = jax.random.randint(minval=0, maxval=self.w,
                                    shape=(self.batch_size,), key=axkey)
            ay = jax.random.randint(minval=0, maxval=self.h,
                                    shape=(self.batch_size,), key=aykey)
        else:
            ax = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.w//2)
            ay = jnp.zeros((self.batch_size,), dtype=int).at[:].set(self.h//2)
        return state.replace(ax=ax, ay=ay,
                             n=0, obs=self.get_obs(ax, ay),
                             rds=jnp.zeros((self.batch_size, self.k,), dtype=bool),
                             rgs=jnp.zeros((self.batch_size, self.k,), dtype=bool),
                             reward=jnp.zeros(self.batch_size),
                             done=jnp.zeros((), dtype=bool))
    

    def visualize(self, state: RoomState, i):
        room = jnp.zeros((self.w, self.h))
        room = room.at[state.rxs[i, :], state.rys[i, :]].set(2)
        room = room.at[state.ax[i], state.ay[i]].set(1) # do we want another color for both, atm overright
        return room

    
    def stack_vis(self, state, env_slice=None):
        if env_slice is None:
            env_slice = jnp.arange(self.batch_size)
            
        room = jnp.zeros((self.w, self.h, 3))
        cd = 1/len(env_slice)
        def stack_up_a(room, xy):
            x, y = xy
            room = room.at[x, y, 1].set(room[x, y, 1] + cd)
            return room, None

        minv, maxv = -2, 2
        def stack_up_ds(room, xyrds):
            xs, ys, rs, ds = xyrds
            good = jnp.where(~ds, jnp.clip(2*(rs-minv)/(maxv-minv)-1, 0, 1), jnp.zeros_like(rs))
            bad  = jnp.where(~ds, jnp.clip(2*(1-(rs-minv)/(maxv-minv))-1, 0, 1), jnp.zeros_like(rs))
            room = room.at[xs, ys, 0].set(room[xs, ys, 0] + cd*good)
            room = room.at[xs, ys, 2].set(room[xs, ys, 2] + cd*bad)
            return room, None
        
        room, _ = jax.lax.scan(stack_up_a, room, (state.ax[env_slice], state.ay[env_slice]))
        room, _ = jax.lax.scan(stack_up_ds, room, (state.rxs[env_slice, :],
                                                   state.rys[env_slice, :],
                                                   state.rrs[env_slice, :],
                                                   state.rds[env_slice, :] & ~state.rgs[env_slice, :]))
        return room


import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera


def save_rollout_multi_gif(env, states, file_name, factor=1):
    fig = plt.figure()
    camera = Camera(fig)
    for state in states:
        room = env.stack_vis(state)**factor
        plt.imshow(np.array(room))
        camera.snap()
    animation = camera.animate()
    animation.save(f"{file_name}.gif", dpi=150, writer="imagemagick")