import argparse

from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---

    parser.add_argument('--num_frames', type=int, default=2e7, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=4, help='number of MDP episodes for adaptation')
    parser.add_argument('--exp_label', default='hyperx', help='label (typically name of method)')
    parser.add_argument('--env_name', default='TinyWorld-v0', help='environment to train on')

    # which exploration bonus(es) to use
    parser.add_argument('--exploration_bonus_hyperstate', type=boolean_argument, default=True, help='bonus on (s, b)')
    parser.add_argument('--exploration_bonus_state', type=boolean_argument, default=False, help='bonus only on (s)')
    parser.add_argument('--exploration_bonus_belief', type=boolean_argument, default=False, help='bonus only on (b)')
    parser.add_argument('--exploration_bonus_vae_error', type=boolean_argument, default=True)

    # --- POLICY ---

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass_state_to_policy', type=boolean_argument, default=True, help='condition policy on state')
    parser.add_argument('--pass_latent_to_policy', type=boolean_argument, default=True,
                        help='condition policy on VAE latent')
    parser.add_argument('--pass_belief_to_policy', type=boolean_argument, default=False,
                        help='condition policy on ground-truth belief')
    parser.add_argument('--pass_task_to_policy', type=boolean_argument, default=False,
                        help='condition policy on ground-truth task description')

    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--policy_state_embedding_dim', type=int, default=32)
    parser.add_argument('--policy_latent_embedding_dim', type=int, default=32)
    parser.add_argument('--policy_belief_embedding_dim', type=int, default=None)
    parser.add_argument('--policy_task_embedding_dim', type=int, default=None)

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm_state_for_policy', type=boolean_argument, default=True, help='normalise state input')
    parser.add_argument('--norm_latent_for_policy', type=boolean_argument, default=True, help='normalise latent input')
    parser.add_argument('--norm_belief_for_policy', type=boolean_argument, default=True, help='normalise belief input')
    parser.add_argument('--norm_task_for_policy', type=boolean_argument, default=True, help='normalise task input')
    parser.add_argument('--norm_rew_for_policy', type=boolean_argument, default=True, help='normalise rew for RL train')
    parser.add_argument('--norm_actions_pre_sampling', type=boolean_argument, default=False,
                        help='normalise policy output')
    parser.add_argument('--norm_actions_post_sampling', type=boolean_argument, default=False,
                        help='normalise policy output')
    parser.add_argument('--norm_rew_clip_param', type=float, default=10, help='rew clip param')

    # network
    parser.add_argument('--policy_layers', nargs='+', default=[64])
    parser.add_argument('--policy_anneal_lr', type=boolean_argument, default=False)

    # PPO specific
    parser.add_argument('--ppo_num_epochs', type=int, default=8, help='number of epochs per PPO update')
    parser.add_argument('--ppo_num_minibatch', type=int, default=4, help='number of minibatches to split the data')
    parser.add_argument('--ppo_clip_param', type=float, default=0.05, help='clamp param')

    # other hyperparameters
    parser.add_argument('--lr_policy', type=float, default=0.0007, help='learning rate (default: 7e-4)')
    parser.add_argument('--num_processes', type=int, default=1,
                        help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--policy_num_steps', type=int, default=100,
                        help='number of env steps to do (per process) before updating')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.1, help='entropy term coefficient')
    parser.add_argument('--policy_gamma', type=float, default=0.98, help='discount factor for rewards')
    parser.add_argument('--policy_tau', type=float, default=0.95, help='gae parameter')
    parser.add_argument('--use_proper_time_limits', type=boolean_argument, default=False,
                        help='treat timeout and death differently')

    # --- VAE TRAINING ---

    # general
    parser.add_argument('--lr_vae', type=float, default=0.001)
    parser.add_argument('--size_vae_buffer', type=int, default=100000,
                        help='how many trajectories (!) to keep in VAE buffer')
    parser.add_argument('--precollect_len', type=int, default=5000,
                        help='how many frames to pre-collect before training begins (useful to fill VAE buffer)')
    parser.add_argument('--vae_batch_num_trajs', type=int, default=25,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--tbptt_stepsize', type=int, default=None,
                        help='stepsize for truncated backpropagation through time; None uses max (horizon of BAMDP)')
    parser.add_argument('--vae_subsample_elbos', type=int, default=None,
                        help='for how many timesteps to compute the ELBO; None uses all')
    parser.add_argument('--vae_subsample_decodes', type=int, default=None,
                        help='number of reconstruction terms to subsample; None uses all')
    parser.add_argument('--vae_avg_elbo_terms', type=boolean_argument, default=False,
                        help='Average ELBO terms (instead of sum)')
    parser.add_argument('--vae_avg_reconstruction_terms', type=boolean_argument, default=False,
                        help='Average reconstruction terms (instead of sum)')
    parser.add_argument('--num_vae_updates', type=int, default=1,
                        help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--kl_weight', type=float, default=0.1, help='weight for the KL term')
    parser.add_argument('--split_batches_by_elbo', type=boolean_argument, default=False,
                        help='split batches up by elbo term (to save memory of if ELBOs are of different length)')

    # - encoder
    parser.add_argument('--action_embedding_size', type=int, default=0)
    parser.add_argument('--state_embedding_size', type=int, default=32)
    parser.add_argument('--reward_embedding_size', type=int, default=8)
    parser.add_argument('--encoder_layers_before_gru', nargs='+', type=int, default=[])
    parser.add_argument('--encoder_gru_hidden_size', type=int, default=128, help='dimensionality of RNN hidden state')
    parser.add_argument('--encoder_layers_after_gru', nargs='+', type=int, default=[])
    parser.add_argument('--latent_dim', type=int, default=10, help='dimensionality of latent space')

    # - decoder: rewards
    parser.add_argument('--decode_reward', type=boolean_argument, default=True, help='use reward decoder')
    parser.add_argument('--rew_loss_coeff', type=float, default=1.0, help='weight for state loss (vs reward loss)')
    parser.add_argument('--input_prev_state', type=boolean_argument, default=False, help='use prev state for rew pred')
    parser.add_argument('--input_action', type=boolean_argument, default=False, help='use prev action for rew pred')
    parser.add_argument('--reward_decoder_layers', nargs='+', type=int, default=[64, 64])

    # - decoder: state transitions
    parser.add_argument('--decode_state', type=boolean_argument, default=False, help='use state decoder')
    parser.add_argument('--state_loss_coeff', type=float, default=1.0, help='weight for state loss')
    parser.add_argument('--state_decoder_layers', nargs='+', type=int, default=[32, 32])

    # - decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    parser.add_argument('--decode_task', type=boolean_argument, default=False, help='use task decoder')
    parser.add_argument('--task_loss_coeff', type=float, default=1.0, help='weight for task loss')
    parser.add_argument('--task_decoder_layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--task_pred_type', type=str, default='task_id', help='choose: task_id, task_description')

    # --- EXPLORATION ---

    # weights for the rewards bonuses
    parser.add_argument('--weight_exploration_bonus_hyperstate', type=float, default=10.0)
    parser.add_argument('--weight_exploration_bonus_state', type=float, default=10.0)
    parser.add_argument('--weight_exploration_bonus_belief', type=float, default=10.0)
    parser.add_argument('--weight_exploration_bonus_vae_error', type=float, default=1.0)
    parser.add_argument('--anneal_exploration_bonus_weights', type=boolean_argument, default=True)

    # hyperparameters for the random network
    parser.add_argument('--rnd_lr', type=float, default=1e-4, help='learning rate ')
    parser.add_argument('--rnd_batch_size', type=int, default=128)
    parser.add_argument('--rnd_update_frequency', type=int, default=1)
    parser.add_argument('--rnd_buffer_size', type=int, default=10000000)
    parser.add_argument('--rnd_output_dim', type=int, default=128)
    parser.add_argument('--rnd_prior_net_layers', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--rnd_predictor_net_layers', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--rnd_norm_inputs', type=boolean_argument, default=False,
                        help='normalise inputs by dividing by var and clipping values')
    parser.add_argument('--rnd_init_weight_scale', type=float, default=10.0,
                        help='by how much to scale the random network weights')

    # other settings
    parser.add_argument('--intrinsic_rew_clip_rewards', type=float, default=10.)
    parser.add_argument('--state_expl_idx', nargs='+', type=int, default=None,
                        help='which part of the state space to do exploration on, None does all')

    # --- ABLATIONS ---

    parser.add_argument('--disable_metalearner', type=boolean_argument, default=False,
                        help='Train feedforward policy')
    parser.add_argument('--add_nonlinearity_to_latent', type=boolean_argument, default=False,
                        help='Use relu before feeding latent to policy')
    parser.add_argument('--disable_decoder', type=boolean_argument, default=False,
                        help='train without decoder')
    parser.add_argument('--rlloss_through_encoder', type=boolean_argument, default=False,
                        help='backprop rl loss through encoder')
    parser.add_argument('--vae_loss_coeff', type=float, default=1.0,
                        help='weight for VAE loss (vs RL loss)')
    parser.add_argument('--condition_policy_on_state', type=boolean_argument, default=True,
                        help='after the encoder, concatenate env state and latent variable')

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log_interval', type=int, default=250, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval, one save per n updates')
    parser.add_argument('--eval_interval', type=int, default=250, help='eval interval, one eval per n updates')
    parser.add_argument('--vis_interval', type=int, default=250, help='visualisation interval, one eval per n updates')
    parser.add_argument('--results_log_dir', default=None, help='directory to save results (None uses ./logs)')

    # general settings
    parser.add_argument('--seed', nargs='+', type=int, default=[73])
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')

    return parser.parse_args(rest_args)
