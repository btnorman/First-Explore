import jax.numpy as jnp
import jax
import jax.random as jandom
from flax import struct
from functools import partial
import matplotlib.pyplot as plt

jax.disable_jit(disable=True)

@partial(jax.jit, static_argnames=('x', 'y', 'pad'))
def aldousbroder(key, x, y, pad=True):
    vedges = jnp.ones((x-1, y)) ## vertical edges
    hedges = jnp.ones((x, y-1)) ## horizontal edges
    visited = jnp.zeros((x,y))

    xk, yk, key = jandom.split(key, 3)
    current = jnp.array((jandom.randint(xk, (), 0, x), jandom.randint(yk, (), 0, y)))
    visited = visited.at[current[0], current[1]].set(1)

    def random_neigh(pos, x, y, key):
        shifts = jnp.array(((-1, 0), (1, 0), (0, -1), (0, 1)), dtype=int)
        neighs = pos + shifts
        valid = (0 <= neighs[:, 0]) & (neighs[:, 0] < x) & (0 <= neighs[:, 1]) & (neighs[:, 1] < y)
        return jandom.choice(key, neighs, p=valid)  

    def cond(carry):
        return ~carry[0].all()
    def step(carry):
        visited, vedges, hedges, pos, key = carry
        n_key, key = jandom.split(key, 2)
        next_cell = random_neigh(pos, x, y, key)

        nvisited = visited.at[next_cell[0], next_cell[1]].set(1)

        xy_to_update = jnp.minimum(pos, next_cell)
        nvedges = jnp.where(pos[0] != next_cell[0],
                            vedges.at[xy_to_update[0], xy_to_update[1]].set(0),
                            vedges)
        nhedges = jnp.where(pos[1] != next_cell[1],
                            hedges.at[xy_to_update[0], xy_to_update[1]].set(0),
                            hedges)
        vedges = jnp.where(visited[next_cell[0], next_cell[1]], vedges, nvedges)
        hedges = jnp.where(visited[next_cell[0], next_cell[1]], hedges, nhedges)
        visited = jnp.where(visited[next_cell[0], next_cell[1]], visited, nvisited)
        carry = (visited, vedges, hedges, next_cell, key)
        return carry

    ans = jax.lax.while_loop(cond, step, (visited, vedges, hedges, current, key))
    vedges, hedges = ans[1:3]
    if pad:
        vedges = jnp.pad(vedges, ((1, 1), (0, 0))).at[0, :].set(1).at[-1, :].set(1)
        hedges = jnp.pad(hedges, ((0, 0), (1, 1))).at[:, 0].set(1).at[:, -1].set(1)
    return vedges.astype(bool), hedges.astype(bool)

@partial(jax.jit, static_argnames=('edge_thickness'))
def shoot_ray(vedges, hedges, ax, ay, angle, edge_thickness=0.05):
    """make an edge only version of the grid"""
    
    # calculates delta y by delta x and visa versa
    dydx = jnp.tan(angle)
    dxdy = (1 / jnp.tan(angle))

    # forward x
    dx = (- edge_thickness - ax) % 1
    fxx_pos = (ax + dx + jnp.arange(0, vedges.shape[0]))
    fxy_pos = (ay + dydx *
              (dx + jnp.arange(0, vedges.shape[0])))
    # backward x
    dx = (ax - edge_thickness) % 1
    bxx_pos = (ax - dx
                  - jnp.arange(0, vedges.shape[0]))
    bxy_pos = (ay - dydx *
                  (dx + jnp.arange(0, vedges.shape[0])))
    
    xx_pos = jnp.where((jnp.pi/2 < angle) & (angle < jnp.pi*3/2), bxx_pos, fxx_pos)
    xy_pos = jnp.where((jnp.pi/2 < angle) & (angle < jnp.pi*3/2), bxy_pos, fxy_pos)
    xi = jnp.round(xx_pos).astype(int)
    yi_min = jnp.floor(xy_pos - edge_thickness).astype(int)
    yi_max = jnp.floor(xy_pos + edge_thickness).astype(int)
    yi_r = jnp.round(xy_pos).astype(int)
    
    x_valid = (xi >= 0) & (xi < vedges.shape[0])
    ymin_valid = (yi_min >= 0) & (yi_min < vedges.shape[1])
    ymax_valid = (yi_max >= 0) & (yi_max < vedges.shape[1])
    yir_valid = (yi_r >= 0) & (yi_r < hedges.shape[1])
    xblock = ((abs(xy_pos - jnp.round(xy_pos)) < edge_thickness) 
              & (hedges[xi-1, yi_r] | hedges[xi, yi_r]))
    x_dists_valid = (x_valid &
                     ((ymin_valid & (vedges[xi, yi_min] == 1))
                      | (ymax_valid & (vedges[xi, yi_max] == 1)))
                      | (yir_valid & xblock))
    x_dists = (xx_pos - ax)**2 + (xy_pos - ay)**2
    i = jnp.where(x_dists_valid, x_dists, jnp.inf).argmin()
    xx, xy, xd = xx_pos[i], xy_pos[i], x_dists[i]
    xd = jnp.where(x_dists_valid.any(), xd, jnp.inf)

    # forward y
    dy = (- edge_thickness - ay) % 1
    fyy_pos = (ay + dy 
          + jnp.arange(0, hedges.shape[1]))
    fyx_pos = (ax + dxdy *
              (dy + jnp.arange(0, hedges.shape[1])))
    # backward y
    dy = (ay - edge_thickness) % 1
    byy_pos = (ay - dy
              - jnp.arange(0, hedges.shape[1]))
    byx_pos = (ax - dxdy *
              (dy + jnp.arange(0, hedges.shape[1])))
    yx_pos = jnp.where(angle > jnp.pi, byx_pos, fyx_pos)
    yy_pos = jnp.where(angle > jnp.pi, byy_pos, fyy_pos)
    yi = jnp.round(yy_pos).astype(int)
    xi_min = jnp.floor(yx_pos - edge_thickness).astype(int)
    xi_max = jnp.floor(yx_pos + edge_thickness).astype(int)
    xi_r = jnp.round(yx_pos).astype(int)

    y_valid = (yi >= 0) & (yi < hedges.shape[1])
    xmin_valid = (xi_min >= 0) & (xi_min < hedges.shape[0])
    xmax_valid = (xi_max >= 0) & (xi_max < hedges.shape[0])
    xir_valid = (xi_r >= 0) & (xi_r < vedges.shape[0])
    yblock = ((abs(yx_pos - jnp.round(yx_pos)) < edge_thickness) 
              & (vedges[xi_r, yi-1] | vedges[xi_r, yi]))
    
    y_dists_valid = (y_valid &
                     ((xmin_valid & (hedges[xi_min, yi] == 1))
                      | (xmax_valid & (hedges[xi_max, yi] == 1))
                      | ((xir_valid & yblock))))

    y_dists = (yx_pos - ax)**2 + (yy_pos - ay)**2
    i = jnp.where(y_dists_valid, y_dists, jnp.inf).argmin()
    yx, yy, yd = yx_pos[i], yy_pos[i], y_dists[i]
    yd = jnp.where(y_dists_valid.any(), yd, jnp.inf)

    tx = jnp.where(xd < yd, xx, yx)
    ty = jnp.where(xd < yd, xy, yy)
    td = jnp.where(xd < yd, xd, yd)
    return tx, ty, td**0.5, xd < yd

# change to have 3 goals

@struct.dataclass
class TinyState:
    ax : jnp.ndarray # agent x
    ay : jnp.ndarray # agent y
    at : jnp.ndarray # agent angle
    gx_min : jnp.ndarray # goal x min
    gx_max : jnp.ndarray # goal x max
    gy_min : jnp.ndarray # goal y min
    gy_max : jnp.ndarray # goal y max
    goal_good : jnp.ndarray # whether the goal is positive or negative 
    vedges : jnp.ndarray # vertical edges
    hedges : jnp.ndarray # horizontal edges
    rds : jnp.ndarray # if reward has been gotten
    cr : jnp.ndarray # cummulative reward
    n : jnp.ndarray
    obs: jnp.ndarray
    obs_extra : jnp.ndarray
    reward: jnp.ndarray
    done : jnp.ndarray

def firstperson_vis(state, env, height, ind=0):
    ray_count = len(env.a)
    hh = height//2
    screen = jnp.zeros((ray_count, height))
    screen = screen.at[:, :hh].set(4)
    m_dists = state.obs[ind, :len(env.a)] * jnp.cos(env.a)    
    shift = (hh*(1/m_dists)/2*jnp.cos(env.a).min()).astype(int)
    shift = shift.clip(0, hh)
    for i in range(ray_count):
        screen = screen.at[i, hh:hh+shift[i]].set(state.obs_extra[ind, i]+1)
        screen = screen.at[i, hh-shift[i]:hh].set(state.obs_extra[ind, i]+1)
    plt.imshow(jnp.swapaxes(screen[::-1, :], 0, 1), aspect=ray_count/height, vmin=0, vmax=4)
    
def birdseye_vis(state, env, ind=0):
    x_vedge = []
    y_vedge = []
    for i in range(state.vedges[ind].shape[0]):
        for j in range(state.vedges[ind].shape[1]):
            if state.vedges[ind][i, j]:
            # if True:
                x_vedge.extend((i, i, None))
                y_vedge.extend((j, j+1, None))            
    x_hedge = []
    y_hedge = []
    for i in range(state.hedges[ind].shape[0]):
        for j in range(state.hedges[ind].shape[1]):
            if state.hedges[ind][i, j]:
            # if True:
                x_hedge.extend((i, i+1, None))
                y_hedge.extend((j, j, None))

    def shoot(tm):
        return shoot_ray(state.vedges[ind], state.hedges[ind],
                         state.ax[ind], state.ay[ind],
                         (state.at[ind]+tm) % (2*jnp.pi))
    tx, ty, td, _ = jax.vmap(shoot)(env.a)
    
    for i in range(len(env.a)):
        plt.plot((state.ax[ind], tx[i]), (state.ay[ind], ty[i]))

    plt.plot((state.gx_min[ind], state.gx_max[ind], state.gx_max[ind],
              state.gx_min[ind], state.gx_min[ind]),
             (state.gy_min[ind], state.gy_min[ind], state.gy_max[ind],
              state.gy_max[ind], state.gy_min[ind]), label='goal')
    plt.plot(x_vedge, y_vedge)
    plt.plot(x_hedge, y_hedge)
    plt.scatter((state.ax[ind],), (state.ay[ind],), label='agent')
    

class TinyWorld:
    def __init__(self, w=3, h=3, side_rays=5, rotation=jnp.pi/4, step_size=0.5,
                 view_field=2*jnp.pi/3, edge_thickness=0.05,
                 ep_len=20,
                 batch_size=128,
                 concat_obs = True,
                 screen_projection=True,
                 type_vis=False,
                 reward_value=1,
                 penalty_value=-1,
                 p_reward=0.3):
        self.w = w
        self.h = h
        self.side_rays = side_rays
        self.view_field = view_field
        self.rotation = rotation
        self.screen_projection = screen_projection
        self.step_size = step_size
        self.ep_len = ep_len
        self.edge_thickness = edge_thickness
        self.batch_size = batch_size
        self.concat_obs = concat_obs
        self.reward_value = reward_value
        self.penalty_value = penalty_value
        self.p_reward = p_reward
        self.type_vis = type_vis
        
        if screen_projection is True:
            self.a = jnp.arctan(jnp.arange(-self.side_rays, self.side_rays+1) 
                   / self.side_rays * jnp.tan(self.view_field/2))
        else:
            self.a = (jnp.arange(-self.side_rays, self.side_rays+1) 
                 / self.side_rays * self.view_field/2)
                   
        vedge_xmins = jnp.zeros((2, 2, 3))
        vedge_xmaxs = jnp.zeros((2, 2, 3))
        vedge_ymins = jnp.zeros((2, 2, 3))
        vedge_ymaxs = jnp.zeros((2, 2, 3))
        hedge_xmins = jnp.zeros((2, 3, 2))
        hedge_xmaxs = jnp.zeros((2, 3, 2))
        hedge_ymins = jnp.zeros((2, 3, 2))
        hedge_ymaxs = jnp.zeros((2, 3, 2))

        for x in range(2):
            for y in range(3):
                vedge_xmins = vedge_xmins.at[0, x, y].set(1+x)
                vedge_xmaxs = vedge_xmaxs.at[0, x, y].set(1+x+0.3)
                vedge_xmins = vedge_xmins.at[1, x, y].set(1+x-0.3)
                vedge_xmaxs = vedge_xmaxs.at[1, x, y].set(1+x)
                vedge_ymins = vedge_ymins.at[:, x, y].set(y)
                vedge_ymaxs = vedge_ymaxs.at[:, x, y].set(y+1)
        for x in range(3):
            for y in range(2):
                hedge_xmins = hedge_xmins.at[:, x, y].set(x)
                hedge_xmaxs = hedge_xmaxs.at[:, x, y].set(x+1)
                hedge_ymins = hedge_ymins.at[0, x, y].set(1+y)
                hedge_ymaxs = hedge_ymaxs.at[0, x, y].set(1+y+0.3)
                hedge_ymins = hedge_ymins.at[1, x, y].set(1+y-0.3)
                hedge_ymaxs = hedge_ymaxs.at[1, x, y].set(1+y)

        xmins = jnp.array((vedge_xmins, jnp.swapaxes(hedge_xmins, -1, -2)))
        xmaxs = jnp.array((vedge_xmaxs, jnp.swapaxes(hedge_xmaxs, -1, -2)))
        ymins = jnp.array((vedge_ymins, jnp.swapaxes(hedge_ymins, -1, -2)))
        ymaxs = jnp.array((vedge_ymaxs, jnp.swapaxes(hedge_ymaxs, -1, -2)))
        assert w == 3
        assert h == 3
        self.xmins = xmins
        self.xmaxs = xmaxs
        self.ymins = ymins
        self.ymaxs = ymaxs

    def _obs(self, vedges, hedges, ax, ay, at,
            gx_min, gx_max, gy_min, gy_max):
        ## vectorized shoot_ray
        def shoot(tm):
            return shoot_ray(vedges, hedges, ax, ay, (at+tm) % (2*jnp.pi),
                             edge_thickness=self.edge_thickness)
        tx, ty, td, ray_type = jax.vmap(shoot)(self.a)
        ongoal = ((tx[:, None] >= gx_min[None, :])
                  & (tx[:, None] <= gx_max[None, :])
                  & (ty[:, None] >= gy_min[None, :])
                  & (ty[:, None] <= gy_max[None, :])).any(axis=-1)
        ray_type = jnp.where(ongoal, 2, ray_type.astype(int))
        if self.concat_obs:
            obs = jnp.concatenate((td, ray_type))
        else:
            obs = td
        return obs, ray_type
    
    
    def meta_reset(self, key):
        key0, key1, key2 = jandom.split(key, 3)
        ax, ay = jnp.zeros((self.batch_size,))+0.5, jnp.zeros((self.batch_size,))+0.5
        at = jnp.zeros((self.batch_size,))+0
        vedges, hedges = jax.vmap(aldousbroder,
                                  in_axes=(0, None, None))(jandom.split(key0, self.batch_size),
                                                           self.w, self.h)
        @jax.vmap
        def choose_goal(vedges, hedges, key):
            p = jnp.zeros((2, 2, 2, 3))
            p = p.at[0, :].set(vedges[1:-1, :])
            p = p.at[1, :].set(hedges[:, 1:-1].T)
            inds = jandom.choice(key, a=len(p.reshape(-1)),
                                 shape=(3,), # 3 goals
                                 p=p.reshape(-1),
                                 replace=False)
            gx_min = self.xmins.reshape(-1)[inds]
            gx_max = self.xmaxs.reshape(-1)[inds]
            gy_min = self.ymins.reshape(-1)[inds]
            gy_max = self.ymaxs.reshape(-1)[inds]
            return gx_min, gx_max, gy_min, gy_max
        
        gx_min, gx_max, gy_min, gy_max = choose_goal(vedges, hedges,
                                                     jandom.split(key1, self.batch_size))
        rds = jnp.zeros((self.batch_size,), dtype=bool)
        n = 0
        done = False
        reward = jnp.zeros((self.batch_size,))
        
        # goal type:
        goal_type = jandom.bernoulli(key2, p=self.p_reward, shape=(self.batch_size, 3))
        
        obs, obs_extra = jax.vmap(self._obs)(vedges, hedges, ax, ay, at,
                                             gx_min, gx_max, gy_min, gy_max)
        return TinyState(ax=ax, ay=ay, at=at,
                         gx_min=gx_min, gx_max=gx_max,
                         gy_min=gy_min, gy_max=gy_max,
                         goal_good=goal_type,
                         vedges=vedges, hedges=hedges,
                         cr = jnp.zeros((self.batch_size,)),
                         rds=rds, n=n, reward=reward, done=done, obs=obs, obs_extra=obs_extra)
        
    def reset(self, state):
        ax, ay = jnp.zeros((self.batch_size,))+0.5, jnp.zeros((self.batch_size,))+0.5
        at = jnp.zeros((self.batch_size,))+0
        rds = jnp.zeros((self.batch_size,), dtype=bool)
        n = 0
        done = False
        reward = jnp.zeros((self.batch_size,))
        obs, obs_extra = jax.vmap(self._obs)(state.vedges, state.hedges, ax, ay, at,
                                             state.gx_min, state.gx_max,
                                             state.gy_min, state.gy_max)

        
        return state.replace(ax=ax, ay=ay, at=at,
                             rds=rds, n=n, reward=reward, done=done,
                             cr = jnp.zeros((self.batch_size,)),
                             obs=obs, obs_extra=obs_extra)
        
    def step(self, state, action):
        # actions: step forward, rotate left, rotate right, no-op

        # step forward
        tx, ty, td, _ = jax.vmap(shoot_ray, in_axes=(0, 0, 0, 0, 0, None)
                            )(state.vedges, state.hedges,
                              state.ax, state.ay,
                              state.at % (2*jnp.pi),
                              self.edge_thickness)
        # valid step forward
        ax = jnp.where((td > self.step_size + 0.1),
                       state.ax + jnp.cos(state.at) * self.step_size,
                       tx - jnp.cos(state.at) * 0.1)
        ay = jnp.where((td > self.step_size + 0.1),
                       state.ay + jnp.sin(state.at) * self.step_size,
                       ty - jnp.sin(state.at) * 0.1)
        ax = jnp.where(action==0, ax, state.ax)
        ay = jnp.where(action==0, ay, state.ay)

        # rotate
        at = jnp.where(action==1, state.at - self.rotation, state.at)
        at = jnp.where(action==2, state.at + self.rotation, at)

        obs, obs_extra = jax.vmap(self._obs)(state.vedges, state.hedges, ax, ay, at,
                                     state.gx_min, state.gx_max,
                                     state.gy_min, state.gy_max)
        
        # ax has batch_dim
        ongoal = ((ax[:, None] >= state.gx_min)
                  & (ax[:, None] <= state.gx_max)
                  & (ay[:, None] >= state.gy_min)
                  & (ay[:, None] <= state.gy_max)) #[batch_dim, 3]
        goal_val = self.reward_value * state.goal_good + self.penalty_value * (1 - state.goal_good)
        goal_val = jnp.where(ongoal.any(axis=-1),
                             (goal_val * ongoal).sum(axis=-1)/ongoal.sum(axis=-1),
                             jnp.zeros(self.batch_size))
        reward = jnp.where(state.rds, jnp.zeros(self.batch_size), goal_val)
        rds = jnp.where(ongoal.any(axis=-1), ongoal.any(axis=-1), state.rds)
        n = state.n + 1
        return state.replace(ax=ax, ay=ay, at=at, obs=obs, obs_extra=obs_extra, n=n, done=n>self.ep_len,
                             cr = state.cr + reward,
                             rds=rds, reward=reward)