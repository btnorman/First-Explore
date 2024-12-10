# Similar to the VariBAD eval python files, loads runs (hacky), performs
# new evaluations (on newly sampled environments) and saves the results


"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--env', type=str)
my_parser.add_argument('--seed', type=int)
args = my_parser.parse_args()

env = args.env
type_ = 'hyperx'

if env == 'bandit':
    env_name = 'StochasticBandit-v0'
elif env == 'mean_bandit':
    env_name = 'StochasticMeanBandit-v0'
elif env == 'control_bandit':
    env_name = 'StochasticControlBandit-v0'
else:
    raise Exception


import numpy as np
import torch
import wandb

# get configs
from config import args_bandit_hyperx

from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner import MetaLearner



def main(in_args):
    parser = argparse.ArgumentParser(in_args)
    parser.add_argument('--env-type', default='room_hyperx')
    parser.add_argument('--wandb_id', type=str, default=None)
    args, rest_args = parser.parse_known_args()
    env = args.env_type
    wandb_id = args.wandb_id
    
    if env == 'bandit' or env == "mean_bandit" or env == 'control_bandit':
        args = args_bandit_hyperx.get_args(rest_args)
        if env == "mean_bandit":
            args.env_name = "StochasticMeanBandit-v0"
        if env == 'control_bandit':
            args.env_name = "StochasticControlBandit-v0"
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

        # start training
        if args.disable_metalearner:
            learner = Learner(args)
        else:
            learner = MetaLearner(args)
    return learner


import os
in_args = ['--env', env, '--wandb_id', 'eval', '--seed', '42',
           '--exp_label', f'eval']
learner = main(in_args)

from utils import helpers as utl
def load_model(learner, model_path):
    learner.policy.actor_critic = torch.load(os.path.join(model_path, f"policy.pt"))
    learner.vae.encoder = torch.load(os.path.join(model_path, f"encoder.pt"))
    if learner.vae.state_decoder is not None:
        learner.vae.state_decoder = torch.load(os.path.join(model_path, f"state_decoder.pt"))
    if learner.vae.reward_decoder is not None:
        learner.vae.reward_decoder = torch.load(os.path.join(model_path, f"reward_decoder.pt"))
    if learner.vae.task_decoder is not None:
        learner.vae.task_decoder = torch.load(os.path.join(model_path, f"task_decoder.pt"))

    # save normalisation params of envs
    if learner.args.norm_rew_for_policy:
        learner.envs.venv.ret_rms = utl.load_obj(model_path, "env_rew_rms")
        
from utils import evaluation as utl_eval

device = 'cpu'
def custom_evaluate(args,
             policy,
             ret_rms,
             iter_idx,
#              tasks,
             encoder=None,
             num_episodes=None
             ):
    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task
    num_processes = args.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    


    # --- initialise environments and latents ---

    envs = make_vec_envs(args=args,
                         seed=args.seed,
                         num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy,
                         ret_rms=ret_rms)
    num_steps = envs._max_episode_steps
    
    actions = torch.zeros((num_processes, num_episodes, num_steps), dtype=int).to(device)
    rewards = torch.zeros((num_processes, num_episodes, num_steps)).to(device)

    # reset environments
    state, belief, task = utl.reset_env(envs, args)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)

    if encoder is not None:
        # reset latent state to prior
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None

    for episode_idx in range(num_episodes):

        for step_idx in range(num_steps):

            with torch.no_grad():
                _, action = utl.select_action(args=args,
                                              policy=policy,
                                              state=state,
                                              belief=belief,
                                              task=task,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar,
                                              deterministic=True)
            actions[range(num_processes), task_count, step_idx] = action.view(-1)
            

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
            done_mdp = [info['done_mdp'] for info in infos]

            if encoder is not None:
                # update the hidden state
                latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(encoder=encoder,
                                                                                              next_obs=state,
                                                                                              action=action,
                                                                                              reward=rew_raw,
                                                                                              done=None,
                                                                                              hidden_state=hidden_state)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)
            rewards[range(num_processes), task_count, step_idx] = rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, indices=done_indices, state=state)

    envs.close()
    return rewards, actions, infos


def get_path(seed):
    path = f"{os.environ['HOME']}/hyperx/hyperx/logs/logs_{env_name}/{type_}_{seed}/models"
    return path


rs_list = []
acts_list = []
task_list = []
load_model(learner, get_path(args.seed))
ret_rms = learner.envs.venv.ret_rms if learner.args.norm_rew_for_policy else None

print(f" env {env} seed {args.seed} evaluatation")
for n in range(625):
    rs, acts, info = custom_evaluate(args=learner.args,
                                     policy=learner.policy,
                                     ret_rms=learner.envs.venv.ret_rms,
                                     encoder=learner.vae.encoder,
                                     iter_idx=learner.iter_idx)
#                                      tasks=learner.train_tasks)
    task = np.array(list(map(lambda x: x['task'], info)))
    rs_list.append(rs)
    acts_list.append(acts)
    task_list.append(task)
    if n % 10 == 0:
        print(n, end=';')

print(f" env {env} seed {args.seed} evaluatation finished")

# bandit shape
rewards = np.array(list(map(lambda x: np.array(x.cpu()),
                            rs_list))).reshape((-1, 100))
actions = np.array(list(map(lambda x: np.array(x.cpu(), dtype=int),
                            acts_list))).reshape((-1, 100))
tasks = np.array(list(map(lambda x: np.array(x, dtype=float),
                          task_list))).reshape((-1, 10))

path = f"{os.environ['HOME']}/hyperx/hyperx/bandit_script/data/rewards_{env}_{args.seed}"
with open(path, 'wb') as file:
    np.save(file, rewards, allow_pickle=False)
path = f"{os.environ['HOME']}/hyperx/hyperx/bandit_script/data/actions_{env}_{args.seed}"
with open(path, 'wb') as file:
    np.save(file, actions, allow_pickle=False)
path = f"{os.environ['HOME']}/hyperx/hyperx/bandit_script/data/tasks_{env}_{args.seed}"
with open(path, 'wb') as file:
    np.save(file, rewards, allow_pickle=False)