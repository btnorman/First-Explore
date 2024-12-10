import argparse
import warnings

import torch

# point robot
from config.pointrobot import \
    args_pointrobot_sparse_hyperx, args_pointrobot_sparse_varibad
# multi-step gridworld
from config.rooms import \
    args_room_hyperx, args_room_varibad, args_room_varibad_x_state
# sparse ant-goal
from config.sparse_ant_goal import \
    args_sparse_ant_goal_rl2, \
    args_sparse_ant_goal_humplik, args_sparse_ant_goal_hyperx, args_sparse_ant_goal_varibad
# sparse cheetah-dir environments
from config.sparse_cheetah_dir import \
    cds_belief_oracle, cds_varibad, cds_hyperx, cds_rl2, cds_humplik
# mountain treasure
from config.treasure_hunt import \
    args_treasure_varibad, args_treasure_hyperx, \
    args_treasure_varibad_x_state, args_treasure_rl2, args_treasure_humplik
from learner import Learner
from metalearner import MetaLearner

from config import args_FE_easy_treasure, args_FE_medi_treasure, args_FE_hard_treasure, args_FE_tiny_world
from config import args_bandit_hyperx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='room_hyperx')
    parser.add_argument('--wandb_id', type=str, default=None)
    args, rest_args = parser.parse_known_args()
    env = args.env_type
    wandb_id = args.wandb_id
    
    # --- First-Explore Tiny World ---
    
    if env == 'fe_tiny_world':
        args = args_FE_tiny_world.get_args(rest_args)
    
    # --- First-Explore Treasure Rooms ---
    
    elif env == 'fe_treasure_easy':
        args = args_FE_easy_treasure.get_args(rest_args)
    elif env == 'fe_treasure_medi':
        args = args_FE_medi_treasure.get_args(rest_args)
    elif env == 'fe_treasure_hard':
        args = args_FE_hard_treasure.get_args(rest_args)
        
    # --- First-Explore Bandit Env ---
    
    elif env == 'bandit' or env == "mean_bandit" or env == 'control_bandit':
        args = args_bandit_hyperx.get_args(rest_args)
        if env == "mean_bandit":
            args.env_name = "StochasticMeanBandit-v0"
        if env == 'control_bandit':
            args.env_name = "StochasticControlBandit-v0"

    # --- Mountain  Treasure ---

    elif env == 'treasure_hunt_varibad':
        args = args_treasure_varibad.get_args(rest_args)
    elif env == 'treasure_hunt_hyperx':
        args = args_treasure_hyperx.get_args(rest_args)
    elif env == 'treasure_hunt_varibad_x_state':
        args = args_treasure_varibad_x_state.get_args(rest_args)
    elif env == 'treasure_hunt_rl2':
        args = args_treasure_rl2.get_args(rest_args)
    elif env == 'treasure_hunt_humplik':
        args = args_treasure_humplik.get_args(rest_args)

    # -- Multi-Stage GridWorld --

    elif env == 'room_varibad':
        args = args_room_varibad.get_args(rest_args)
    elif env == 'room_varibad_x_state':
        args = args_room_varibad_x_state.get_args(rest_args)
    elif env == 'room_hyperx':
        args = args_room_hyperx.get_args(rest_args)

    # --- Sparse MUJOCO Half Cheetah ---

    elif env == 'cds_belief_oracle':
        args = cds_belief_oracle.get_args(rest_args)
    elif env == 'cds_varibad':
        args = cds_varibad.get_args(rest_args)
    elif env == 'cds_hyperx':
        args = cds_hyperx.get_args(rest_args)
    elif env == 'cds_rl2':
        args = cds_rl2.get_args(rest_args)
    elif env == 'cds_humplik':
        args = cds_humplik.get_args(rest_args)

    # --- Sparse MUJOCO Ant Goal ---

    elif env == 'sparse_ant_goal_rl2':
        args = args_sparse_ant_goal_rl2.get_args(rest_args)
    elif env == 'sparse_ant_goal_varibad':
        args = args_sparse_ant_goal_varibad.get_args(rest_args)
    elif env == 'sparse_ant_goal_humplik':
        args = args_sparse_ant_goal_humplik.get_args(rest_args)
    elif env == 'sparse_ant_goal_hyperx':
        args = args_sparse_ant_goal_hyperx.get_args(rest_args)

    # --- Sparse Point Robot ---

    elif env == 'pointrobot_sparse_varibad':
        args = args_pointrobot_sparse_varibad.get_args(rest_args)
    elif env == 'pointrobot_sparse_hyperx':
        args = args_pointrobot_sparse_hyperx.get_args(rest_args)

    else:
        raise NotImplementedError

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down.')

    # check if we're adding an exploration bonus
    args.add_exploration_bonus = args.exploration_bonus_hyperstate or \
                                 args.exploration_bonus_state or \
                                 args.exploration_bonus_belief or \
                                 args.exploration_bonus_vae_error

    # clean up arguments
    if hasattr(args, 'disable_decoder') and args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    # loop through all passed seeds
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        
        args.seed = seed
        args.action_space = None
        
        config = dict(args.__dict__)
        config.update({'env_type' : env})
        
        wandb.init(project='hyperx', id=str(seed)+str(wandb_id), resume="allow",
                   sync_tensorboard=True, config = config)
        print('training', seed)


        # start training
        if args.disable_metalearner:
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
        learner.train()
        wandb.finish()


if __name__ == '__main__':
    main()
