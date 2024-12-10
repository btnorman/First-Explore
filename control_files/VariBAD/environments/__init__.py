from gym.envs.registration import register



# # First-Explore Tiny World Environment
register(
    'EasyWorld-v0',
    entry_point='environments.tiny_world:TinyWorldEnv',
    kwargs={'p_reward' : 0.3, 'penalty_value' : 0,}
)

register(
    'HardWorld-v0',
    entry_point='environments.tiny_world:TinyWorldEnv',
    kwargs={'p_reward' : 0.3, 'penalty_value' : -1,}
)

# Mujoco
# ----------------------------------------

# - randomised reward functions

register(
    'AntDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntDir2D-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
            'max_episode_steps': 200},
    max_episode_steps=200,
)

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HumanoidDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

# - randomised dynamics

register(
    id='Walker2DRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
    max_episode_steps=200
)

register(
    id='HopperRandParams-v0',
    entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
    max_episode_steps=200
)


# # 2D Navigation
# # ----------------------------------------
#
register(
    'PointEnv-v0',
    entry_point='environments.navigation.point_robot:PointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100,
            'goal_sampler': 'semi-circle'
            },
    max_episode_steps=100,
)

#
# # GridWorld
# # ----------------------------------------

register(
    'GridNavi-v0',
    entry_point='environments.navigation.gridworld:GridNavi',
    kwargs={'num_cells': 5, 'num_steps': 15},
)

#
# # First-Explore Treasure-Room
# # ---------------------------------

register(
    'HardTreasureEnv-v0',
    entry_point='environments.treasure_env:TreasureEnv',
    kwargs={'minval': -4,},
)

register(
    '3TreasureEnv-v0',
    entry_point='environments.treasure_env:TreasureEnv',
    kwargs={'minval': -2*3**0.5,},
)

register(
    '2TreasureEnv-v0',
    entry_point='environments.treasure_env:TreasureEnv',
    kwargs={'minval': -2*2**0.5,},
)

register(
    'EasyTreasureEnv-v0',
    entry_point='environments.treasure_env:TreasureEnv',
    kwargs={'minval': 0,},
)

register(
    'MediTreasureEnv-v0',
    entry_point='environments.treasure_env:TreasureEnv',
    kwargs={'minval': -3,},
)

#x = 2 x sqrt(2) (needs two exploits to make even)
# x = 2 x sqrt(3) (needs three exploits to make even)
# x = 2 x sqrt(4) (needs four exploits to make even)d



#
# # Stochastic Bandit
# # -------------------------------

register(
    'StochasticBandit-v0',
    entry_point='environments.bandit_env:BanditEnv',
)

register(
    'StochasticControlBandit-v0',
    entry_point='environments.bandit_env:MeanBanditEnv',
    kwargs={'minval': 0,}
)

register(
    'StochasticMeanBandit-v0',
    entry_point='environments.bandit_env:MeanBanditEnv',
    kwargs={'minval': 0.5,}
)