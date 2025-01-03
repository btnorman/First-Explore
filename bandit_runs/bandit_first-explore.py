### script set up

import os  # ensure we can import the lte_code
import sys
sys.path.append(os.getcwd()+"/..")
sys.path

from prep_args import C
import prep_args

script_setting = {'arm_num':C((10,)),
                  'pulls':C((100,)),
                  'batch_size' : C((128,)),
                  'bandit_type' : C(("mean", "control")),
                  'train_len' : C((200,)),
                  'seed' : C((4, 5, 6, 7, 8))}

# Arm number
# Context Length
# Batch Size
# Bandit Type
# Train Epochs

n = prep_args.count(script_setting)
args = prep_args.get_args(n)
k = args.SBATCHID
prep_args.set_run_parameters(script_setting, k)

ACT_DIM = script_setting['arm_num']
SEQ_LEN = script_setting['pulls']
BATCH_SIZE = script_setting['batch_size']

run_name = f"n{ACT_DIM}_p{SEQ_LEN}_b{BATCH_SIZE}_{script_setting['bandit_type']}_seed{script_setting['seed']}" #NOTE: does not include train_len
checkpoint_dir = run_name

### env setup

from lte_code.bandit import Bandit, MeanBandit
import jax
import jax.numpy as jnp
ACT_DIM = script_setting['arm_num']

from lte_code.lte_model3 import LTE
from transformers import DecisionTransformerConfig


num_training_steps = 1000*script_setting['train_len']

import optax
def warmup_linear_schedule(
    learning_rate: float,
    total_steps: int,
    warmup_ratio: int):
    warmup_steps = int(total_steps*warmup_ratio)
    schedules = [
      optax.linear_schedule(
          init_value=0,
          end_value=learning_rate,
          transition_steps=warmup_steps),
      optax.linear_schedule(
          init_value=learning_rate,
          end_value=0,
          transition_steps= total_steps - warmup_steps),]
    return optax.join_schedules(schedules, [warmup_steps])


schedule = warmup_linear_schedule(learning_rate=1e-4,
                                  total_steps=num_training_steps,
                                  warmup_ratio=0.1)
optimizer = optax.chain(
  optax.clip(0.25),
  optax.adamw(learning_rate=schedule, weight_decay=1e-4),
)


config = DecisionTransformerConfig(act_dim=ACT_DIM, state_dim=ACT_DIM, n_head=4)
model = LTE(config)
key = jax.random.PRNGKey(script_setting['seed'])
pkey, key = jax.random.split(key)

if script_setting['bandit_type'] == 'normal':
    ENV = Bandit(key=jax.random.PRNGKey(42), n=ACT_DIM,
                     deterministic=False, noise_scale=0.5)
elif script_setting['bandit_type'] == 'RL2':
    ENV = RL2_Bandit(key=jax.random.PRNGKey(42), n=ACT_DIM)
elif script_setting['bandit_type'] == 'mean':
    ENV = MeanBandit(key=jax.random.PRNGKey(42), n=ACT_DIM, minval=0.5)
elif script_setting['bandit_type'] == 'control':
    ENV = MeanBandit(key=jax.random.PRNGKey(42), n=ACT_DIM, minval=0)

batch_step = jax.vmap(ENV.step)
batch_mset = jax.vmap(ENV.meta_reset)

if script_setting['bandit_type'] == 'normal':
    maxi = jax.random.normal(key=key, shape=(10000, ACT_DIM)).max(axis=1).mean()
elif script_setting['bandit_type'] == 'RL2':
    maxi = jax.random.uniform(key=key, shape=(10000, ACT_DIM)).max(axis=1).mean()
elif script_setting['bandit_type'] == 'mean':
    maxi = (jax.random.normal(key=key, shape=(10000, ACT_DIM))).at[:, 0].set(0.5).max(axis=1).mean()
elif script_setting['bandit_type'] == 'control':
    maxi = (jax.random.normal(key=key, shape=(10000, ACT_DIM))).at[:, 0].set(0).max(axis=1).mean()


def reward_sequence(states, actions):
    """takes a batch of states
    and a batch of actions [batch_dim, seq_len, act_dim]
    and calculates the rwards, [batch_dim, seq_len, rewards]"""
    def step_state_reward(state, action):
        state = batch_step(state, action)
        return state, state.reward
    states, rewards = jax.lax.scan(step_state_reward,
                                   states,
                                   jnp.swapaxes(actions, 0, 1))
    return jnp.swapaxes(rewards, 0, 1)[..., 0]

def max_in_seq(values):
    """takes a batch of values [batch_dim, seq_len]
    and calculates the the running maximums along the seq_len"""
    def run_max(max_, val):
        max_ = jnp.maximum(max_, val)
        return max_, max_
    return jnp.swapaxes(jax.lax.scan(run_max, -jnp.inf+jnp.zeros(values.shape[0]),
                        jnp.swapaxes(values, 0, 1))[1],
                        0, 1)

def tokenize(actions_batch, rewards_batch):
    @jax.vmap
    @jax.vmap
    def one_hot(act):
        return jnp.zeros(shape=ACT_DIM).at[act].set(1)
    action_tokens = one_hot(actions_batch)
    reward_tokens = jnp.expand_dims(rewards_batch, 2)
    return action_tokens, reward_tokens

import flax
def epsilon_samp(logits, e):
    p = flax.linen.activation.softmax(logits)
    return jnp.log((1-e)*p+e*jnp.ones(logits.shape)/logits.shape[-1])


from functools import partial

@partial(jax.jit, static_argnames=('argmax', 'epsilon', 'greater'))
def exploit(model_params, states, actions, rewards, key,
            running_max, argmax=False, greater=False,
            epsilon=0):
    """Exploiting at each step of the sequence
    return the rewards, and the loss"""    
    action_tokens, reward_tokens = tokenize(actions_batch=actions,
                                            rewards_batch=rewards)
    batch_size, seq_len, act_dim = action_tokens.shape
    action_tokens = jnp.concatenate((jnp.zeros((batch_size, 1, act_dim)),
                                     action_tokens), axis=1)
    reward_tokens = jnp.concatenate((jnp.zeros((batch_size, 1, 1)),
                                     reward_tokens), axis=1)

    time_steps = jnp.zeros((batch_size, seq_len+1), dtype=int).at[:].set(jnp.arange(seq_len+1))
    hidden_state = model.apply(model_params,
                               actions=action_tokens,
                               rewards=reward_tokens,
                               position_ids=time_steps).last_hidden_state
    max_logits = model.pred_max.apply(model_params['pred_max'],
                                      hidden_state)
    nonmax_logits = model.pred_nonmax.apply(model_params['pred_nonmax'],
                                            hidden_state)
    ckey, ukey, rkey = jax.random.split(key, 3)
    if epsilon > 0:
        sample_logits = jax.lax.stop_gradient(epsilon_samp(max_logits, epsilon))
    else:
        sample_logits = jax.lax.stop_gradient(max_logits)
    if not argmax:
        m_actions = jax.random.categorical(logits=sample_logits, key=ckey, axis=-1)
    else:
        m_actions = jnp.argmax(max_logits, axis=-1)

    m_rewards = reward_sequence(states, m_actions)
    running_max = jnp.append((jnp.zeros((batch_size,1))-jnp.inf), running_max, axis=1)
    
    if greater:
        action_preds = jnp.where(jnp.expand_dims(m_rewards > running_max, 2),
                                 max_logits, nonmax_logits)
    else:
        action_preds = jnp.where(jnp.expand_dims(m_rewards >= running_max, 2),
                                 max_logits, nonmax_logits)
    ## important sanity step here stopping the gradients
    loss = optax.softmax_cross_entropy_with_integer_labels(action_preds+sample_logits,
                                                           jax.lax.stop_gradient(m_actions)).mean()
    ## in general, if we are sampling with logits_a
    ## and we are estimating logits_b
    ## then we want softmax(logits_a+logits_b, seen stuff)
    return m_rewards, loss


def exploit_data_test(model_params, key, batch_size, epsilon):
    m_key, a_key, e_key = jax.random.split(key, 3)
    states = batch_mset(jax.random.split(m_key, batch_size))
    actions = jax.random.randint(minval=0, maxval=ACT_DIM,
                                 shape=(batch_size, SEQ_LEN-1),
                                 key=a_key)
    rewards = reward_sequence(states=states,
                              actions=actions)
    running_max = max_in_seq(rewards)

    return exploit(model_params, states, actions, rewards, e_key,
                              running_max,
                              epsilon=epsilon)[1]


@jax.jit
def train_step(carry, _):
    params, opt_state, key = carry
    next_key, data_key, drop_key = jax.random.split(key, 3)
    loss, grad = jax.value_and_grad(exploit_data_test)(params,
                                                       data_key, 128,
                                                       0.5)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return (params, opt_state, next_key), loss

@jax.jit
def eval_step(carry, _):
    params, opt_state, key = carry
    next_key, data_key, drop_key = jax.random.split(key, 3)
    loss, grad = jax.value_and_grad(exploit_data_test)(params,
                                                       data_key, 128,
                                                       0.5)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return (params, opt_state, next_key), loss



def run_exploit(model_params,
                states,
                actions,
                key, argmax=True,epsilon=0):
    m_key, a_key, e_key = jax.random.split(key, 3)
    rewards = reward_sequence(states=states,
                              actions=actions)
    running_max = max_in_seq(rewards)
    return exploit(model_params, states, actions, rewards, e_key,
                   running_max,
                   argmax=argmax,
                   epsilon=epsilon)[0]

import matplotlib.pyplot as plt

batch_size = 10000
exhaustive = jnp.repeat(jnp.expand_dims(jnp.repeat(jnp.expand_dims(jnp.arange(ACT_DIM),
                                                      1),
                                      (SEQ_LEN//ACT_DIM)+1, axis=1).T.reshape(-1)[:SEQ_LEN-1],
                           0),
           batch_size, axis=0)
random = jax.random.randint(minval=0, maxval=ACT_DIM,
                           shape=(batch_size, SEQ_LEN-1), key=key)


def save_exploit_plot(plot_params, save_name, title):
    batch_size=1000
    bstate = batch_mset(jax.random.split(key, batch_size))
    plt.clf()
    plt.plot(run_exploit(plot_params,
            bstate,
            exhaustive[:batch_size],
            key, argmax=True).mean(axis=0).reshape(-1), label="exhaustive explore\nargmax exploit")
    plt.plot(run_exploit(plot_params,
            bstate,
            exhaustive[:batch_size],
            key, argmax=False).mean(axis=0).reshape(-1), label="exhaustive explore\nsampling exploit")
    plt.plot(run_exploit(plot_params,
            bstate,
            random[:batch_size],
            key, argmax=True).mean(axis=0).reshape(-1), label="random explore\nargmax exploit")
    plt.plot(run_exploit(plot_params,
            bstate,
            random[:batch_size],
            key, argmax=False).mean(axis=0).reshape(-1), label="random explore\nsampling exploit")
    plt.title(title)
    plt.xlabel("context length, $k$, provided")
    plt.ylabel("average reward")
    plt.xticks(jnp.arange(SEQ_LEN))
    ax = plt.gca()
    ax.set_ylim([-0.1, maxi+0.1])
    plt.legend()
    plt.grid()
    plt.savefig(save_name, bbox_inches='tight')
    
    
### Functions for autoregressive sampling

from flax.core.frozen_dict import unfreeze, freeze
def feed_token(model_params, cache,
               actions, rewards, position_ids):
    """takes the parameters, the current model cache,
    the token, token_type, and time_step
    and feeds them to the model updating the cache
    
    Note, can process multiple tokens at once
    token should be [batch_size, seq_len, token_dim]
    and token_type [batch_size, seq_len]
    time_step [batch_size, seq_len]"""
    ra = unfreeze(model_params)
    ra['seq']['cache'] = cache['cache']
    return model.apply(params=ra, mutable=['cache'],
                       actions=actions,
                       rewards=rewards,
                       position_ids=position_ids)

from functools import partial
@partial(jax.jit, static_argnames="batch_size")
def init_cache(model_params, batch_size):
    return model.apply(
        params=model_params,
        init_cache=True,
        mutable=['cache'],
        actions=jnp.zeros((batch_size, SEQ_LEN, ACT_DIM)),
        rewards=jnp.zeros((batch_size, SEQ_LEN, 1)),
        position_ids=jnp.zeros((batch_size, SEQ_LEN), dtype=int),
    )[1]

# batch_size = 128
# CACHE = freeze(init_cache(new_params, batch_size))


@partial(jax.jit, static_argnames=("batch_size", "argmax", "pred_f1", "pred_f2", "pred_f3"))
def autoregressive_rollout(model_params, key, batch_size, state,
                           pred_f1, pred_f2, pred_f3, argmax=False):
    """function to do a rollout given a policy
    
    model_params specifies the model parameters
    key is the jax.random.PRNGKey to seed the rng
    batch_size is batch_size
    state is the initial environment state
    and pred_f is the function that maps 
    the model hidden state to action probabilities
    argmax is whether to select the most probable action
    or to sample
    and epsilon is the probability of random action selection"""
    cache = init_cache(model_params, batch_size)
    act_token = jnp.zeros((batch_size, 1, ACT_DIM))
    reward_token = jnp.zeros((batch_size, 1, 1))
    time_step = jnp.zeros((batch_size, 1))
    m_key, a_key, a2_key = jax.random.split(key, 3)
    cache = init_cache(model_params, batch_size)
    carry = act_token, reward_token, time_step, cache, state

    def one_step(carry, key):
        act_token, reward_token, time_step, cache, state = carry
        ans, cache = feed_token(model_params=model_params,
                                cache=cache,
                                actions=act_token,
                                rewards=reward_token, 
                                position_ids=time_step)
        hidden_state = ans.last_hidden_state
        # batch_size, seq_len, act_dim
        logits = pred_f1(hidden_state[:, -1, ...])
        logits_2 = pred_f2(hidden_state[:, -1, ...])
        logits_3 = pred_f3(hidden_state[:, -1, ...])
        ckey, ukey, rkey = jax.random.split(key, 3)
        if not argmax:
            actions = jax.random.categorical(logits=logits,
                                             key=ckey, axis=-1)
        else:
            actions = jnp.argmax(logits, axis=-1)
        state = batch_step(state, actions)
        
        @jax.vmap
        def one_hot(act):
            return jnp.zeros(shape=ACT_DIM).at[act].set(1)
        act_token = jnp.expand_dims(one_hot(actions), 1)
        reward_token = jnp.expand_dims(state.reward, 1)
        time_step = time_step+1
        return ((act_token, reward_token, time_step, cache, state),
                (actions, state.reward[:, 0], logits, logits_2, logits_3))
    carry, x = one_step(carry, a_key)
    _, xs = jax.lax.scan(one_step, carry, 
                         jax.random.split(a2_key, SEQ_LEN-2))
    xs = jax.tree_map(lambda a, b :jnp.swapaxes(jnp.append(jnp.expand_dims(a, 0),
                                                                         b, axis=0),
                                                              0, 1), 
                                    x, xs)
    return xs



def explore(model_loit_params,
            model_lore_params,
            states, batch_size, key, e_epsilon=0, m_epsilon=0,
           e_argmax=False, m_argmax=False, new_explore=script_setting['new_explore']):
    """explores autoregressively from the states
    and calcualtes the exploit and explore loss"""
    
    rollout_key, exploit_key = jax.random.split(key, 2)
    ### generate the explore sequence
    
#     @jax.jit
    def predf1(hidden_state): 
        logits = model.pred_exp.apply(model_lore_params['pred_exp'],
                                              hidden_state)
        logits = jax.lax.stop_gradient(epsilon_samp(logits, e_epsilon))
        return logits
    
#     @jax.jit
    def predf2(hidden_state): 
        logits = model.pred_nonexp.apply(model_lore_params['pred_exp'],
                                         hidden_state)
        return logits
    
    def predf3(hidden_state): 
        logits = model.pred_nonexp.apply(model_lore_params['pred_nonexp'],
                                         hidden_state)
        return logits

    actions, rewards, sample_logits, logits2, logits3 =\
    autoregressive_rollout(model_params=model_lore_params,
                           key=rollout_key,
                           batch_size=batch_size,
                           state=states,
                           pred_f1=predf1,
                           pred_f2=predf2,
                           pred_f3=predf3,
                           argmax=e_argmax)
    running_max = max_in_seq(rewards)
    loit_rewards, loit_loss =\
    exploit(model_params=model_loit_params,
            states=states,
            actions=actions,
            rewards=rewards,
            key=exploit_key,
            running_max=running_max,
            argmax=m_argmax,
            epsilon=m_epsilon)
    
    explore_cond = loit_rewards[:, 1:] > jnp.pad(running_max[:, :-1],
                                                 ((0, 0), (1,0)),
                                                 constant_values=-jnp.inf)
    action_preds = jnp.where(jnp.expand_dims(explore_cond, 2),
                             logits2, logits3)
    lore_loss = optax.softmax_cross_entropy_with_integer_labels(\
    action_preds+sample_logits, jax.lax.stop_gradient(actions)).mean()
    return loit_rewards, loit_loss, lore_loss # exploit rewards, exploit loss, explore loss



def explore_train(model_loit_params,
                  model_lore_params,
                  key, batch_size, e_epsilon, m_epsilon):
    m_key, a_key, e_key = jax.random.split(key, 3)
    states = batch_mset(jax.random.split(m_key, batch_size))
    rewards, loit_loss, lore_loss =\
    explore(model_loit_params=model_loit_params,
            model_lore_params=model_lore_params,
            states=states, batch_size=batch_size,
            key=key, e_epsilon=e_epsilon, m_epsilon=m_epsilon)
    return loit_loss, lore_loss

def explore_train_loit_grad(model_lore_params,
                            model_loit_params,
                            key, batch_size,
                            e_epsilon,
                            m_epsilon):
    (loit_loss, lore_loss), loit_grad = jax.value_and_grad(explore_train,
                                                           has_aux=True)\
        (model_loit_params,
         model_lore_params,
         key, batch_size, e_epsilon, m_epsilon)
    return lore_loss, (loit_loss, loit_grad)

def explore_losses_and_grads(model_loit_params,
                   model_lore_params,
                   key, batch_size,
                   e_epsilon,
                   m_epsilon):
    (lore_loss, (loit_loss, loit_grad,)), lore_grad = jax.value_and_grad(explore_train_loit_grad,
                                                                      has_aux=True)\
        (model_lore_params,
         model_loit_params,
         key, batch_size,
         e_epsilon,
         m_epsilon)
    return loit_loss, loit_grad, lore_loss, lore_grad


@jax.jit
def train_e_step(carry, _):
    (model_params, opt_state, key) = carry
    next_key, data_key, drop_key = jax.random.split(key, 3)
    loit_loss, loit_grad, lore_loss, lore_grad = explore_losses_and_grads(model_params,
                                                                          model_params,
                                          data_key, 128,
                                          0.05, 0.05)
    grad = jax.tree_map(lambda a, b: a+b, loit_grad, lore_grad)
    updates, opt_state = optimizer.update(grad, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)
    return (model_params, opt_state, next_key), (loit_loss, lore_loss)

@jax.jit
def eval_e_step(carry, _):
    (model_params, opt_state, key) = carry
    next_key, data_key, drop_key = jax.random.split(key, 3)
    loit_loss, loit_grad, lore_loss, lore_grad = explore_losses_and_grads(model_params,
                                                                          model_params,
                                          data_key, 128,
                                          0.05, 0.05)
    grad = jax.tree_map(lambda a, b: a+b, loit_grad, lore_grad)
    updates, opt_state = optimizer.update(grad, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)
    return (model_params, opt_state, next_key), (loit_loss, lore_loss)



def explore__exploit_rewards(loit_params, lore_params, key, batch_size, e_argmax, m_argmax):
    m_key, e_key = jax.random.split(key, 2)
    states = batch_mset(jax.random.split(m_key, batch_size))
    rewards, loit_loss, lore_loss = explore(loit_params,
                                            lore_params,
                                           states,
                                           batch_size,
                                           key=e_key,
                                           e_epsilon=0,
                                           m_epsilon=0,
                                           e_argmax=e_argmax,
                                           m_argmax=m_argmax)
    return rewards


import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, IndexLocator)


def save_ee_plot(loit_params, lore_params, save_name, title):
    batch_size=1000
    bstate = batch_mset(jax.random.split(key, batch_size))
    plt.clf()
    plt.plot(run_exploit(loit_params,
            bstate,
            exhaustive[:batch_size],
            key, argmax=True).mean(axis=0).reshape(-1), label="e-explore, a-max")
    plt.plot(run_exploit(loit_params,
            bstate,
            exhaustive[:batch_size],
            key, argmax=False).mean(axis=0).reshape(-1), label="e-explore, s-max")
    plt.plot(run_exploit(loit_params,
            bstate,
            random[:batch_size],
            key, argmax=True).mean(axis=0).reshape(-1), label="r-explore, a-max")
    plt.plot(run_exploit(loit_params,
            bstate,
            random[:batch_size],
            key, argmax=False).mean(axis=0).reshape(-1), label="r-explore, s-max")
    em_ea_ma = explore__exploit_rewards(loit_params,
                                        lore_params, key, batch_size, e_argmax=True, m_argmax=True)
    em_ea_ms = explore__exploit_rewards(loit_params,
                                        lore_params, key, batch_size, e_argmax=True, m_argmax=False)
    em_es_ma = explore__exploit_rewards(loit_params,
                                        lore_params, key, batch_size, e_argmax=False, m_argmax=True)
    em_es_ms = explore__exploit_rewards(loit_params,
                                        lore_params, key, batch_size, e_argmax=False, m_argmax=False)
    plt.plot(em_ea_ma.mean(axis=0).reshape(-1), label="a-explore, a-max")
    plt.plot(em_ea_ms.mean(axis=0).reshape(-1), label="a-explore, s-max")
    plt.plot(em_es_ma.mean(axis=0).reshape(-1), label="s-explore, a-max")
    plt.plot(em_es_ms.mean(axis=0).reshape(-1), label="s-explore, s-max")
    plt.title(title)
    plt.xlabel("context length, $k$, provided")
    plt.ylabel("average reward")
    plt.xticks(jnp.arange(1, SEQ_LEN, 5))
    ax = plt.gca()
    ax.set_ylim([0, ACT_DIM])
    ax.xaxis.set_minor_locator(IndexLocator(1, 0))
    ax.xaxis.grid(True, which='minor')
    ax.set_ylim([-0.1, maxi+0.1])
    plt.legend()
    plt.grid()
    plt.savefig(save_name, bbox_inches='tight')

    
@partial(jax.jit)
def num_u(arr, k):
    ans = 0
    for i in range(ACT_DIM):
        a = jnp.swapaxes((arr[:, :] == i+jnp.zeros(arr.shape[1])), 0, 1)
        ans += jnp.where((jnp.arange(arr.shape[1]) <= k)[:, None], a, jnp.zeros(1)).any(axis=0).mean()
    return ans

def prog(arr):
    return jax.vmap(lambda k: num_u(arr, k))(jnp.arange(SEQ_LEN-1))


@partial(jax.jit, static_argnames=("batch_size", "argmax"))
def explore_acts(lore_params, key, batch_size, bstate, argmax):
    def predf1(hidden_state): 
        logits = model.pred_exp.apply(lore_params['pred_exp'],
                                              hidden_state)
        return logits
   
    def predf2(hidden_state): 
        logits = model.pred_nonexp.apply(lore_params['pred_exp'],
                                         hidden_state)
        return logits
    
    def predf3(hidden_state): 
        logits = model.pred_nonexp.apply(lore_params['pred_nonexp'],
                                         hidden_state)
        return logits
    return autoregressive_rollout(lore_params, key, batch_size, bstate,
                                  predf1, predf2, predf3, argmax=argmax)[0]


@partial(jax.jit, static_argnames=("batch_size", "argmax"))
def exploit_acts(lore_params, key, batch_size, bstate, argmax):
    def predf1(hidden_state): 
        logits = model.pred_exp.apply(lore_params['pred_max'],
                                              hidden_state)
        return logits
   
    def predf2(hidden_state): 
        logits = model.pred_nonexp.apply(lore_params['pred_exp'],
                                         hidden_state)
        return logits
    
    def predf3(hidden_state): 
        logits = model.pred_nonexp.apply(lore_params['pred_nonexp'],
                                         hidden_state)
        return logits
    return autoregressive_rollout(lore_params, key, batch_size, bstate,
                                  predf1, predf2, predf3, argmax=argmax)[0]


random_coverage = prog(random)
def coverage_plot(lore_params, save_name, title):
    batch_size=1000
    bstate = batch_mset(jax.random.split(key, batch_size))
    a_lore = explore_acts(lore_params, key, batch_size, bstate, argmax=True)
    a_lore_c = prog(a_lore)
    s_lore = explore_acts(lore_params, key, batch_size, bstate, argmax=False)
    s_lore_c = prog(s_lore)
    a_loit = exploit_acts(lore_params, key, batch_size, bstate, argmax=True)
    a_loit_c = prog(a_loit)
    s_loit = exploit_acts(lore_params, key, batch_size, bstate, argmax=False)
    s_loit_c = prog(s_loit)
    plt.clf()
    plt.plot(jnp.arange(1, SEQ_LEN), a_lore_c, label="explore argmax")
    plt.plot(jnp.arange(1, SEQ_LEN), s_lore_c, label="explore sampling")
    plt.plot(jnp.arange(1, SEQ_LEN), a_loit_c, label="exploit argmax")
    plt.plot(jnp.arange(1, SEQ_LEN), s_loit_c, label="exploit sampling")
    plt.plot(jnp.arange(1, SEQ_LEN), random_coverage, label="random sampling")
#     plt.plot(jnp.arange(1, 20), exhaustive, label="random sampling")
    plt.title(title)
    plt.xlabel("number of actions")
    plt.ylabel("average # of unique actions")
    plt.xticks(jnp.arange(1, SEQ_LEN, 5))
    ax = plt.gca()
    ax.set_ylim([0, ACT_DIM])
    ax.xaxis.set_minor_locator(IndexLocator(1, 0))
    ax.xaxis.grid(True, which='minor')
    plt.legend()
    plt.grid()
    plt.savefig(save_name, bbox_inches='tight')



### Checkpointing Logic:


import wandb
wandb.init(project="LTE", id=args.SLURM_JOB_ID, resume="allow")
wandb.config.update({'script_setting' : script_setting})
import pickle
import os


key1, key2 = jax.random.split(key, 2)
new_params = model.init(key1)
opt_state = optimizer.init(new_params)
next_carry = (new_params, opt_state, key)
epoch = 0



if not os.path.isdir(checkpoint_dir):
    if os.path.exists(checkpoint_dir):
        raise Exception("the checkpoint directory is not a directory!")
    else:
        os.makedirs(checkpoint_dir)
path = os.path.join(checkpoint_dir, 'run_data.pkl')
if os.path.exists(path):
    with open(path, 'rb') as file:
        next_carry, epoch = pickle.load(file)
        print("loaded checkpoint")
else:
    print("no checkpoint found, initializing instead")  


def save(name, carry, epoch):
    if not os.path.isdir(checkpoint_dir):
        if os.path.exists(checkpoint_dir):
            raise Exception("the checkpoint directory is not a directory!")
        else:
            os.makedirs(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, f'tmp_{name}.pkl'), 'wb') as file:
        pickle.dump((carry, epoch), file=file)
    os.replace(os.path.join(checkpoint_dir, f'tmp_{name}.pkl'),
               os.path.join(checkpoint_dir, f'{name}.pkl'))


while epoch < script_setting['train_len']:
    next_carry, (loit_losses, lore_losses) = jax.lax.scan(train_e_step,
                                      next_carry, None, length=1000)
    _, (loit_eval, lore_eval) = eval_e_step(next_carry, None)
    
    wandb.log({"loit_losses" : loit_losses.mean(),
               "loit_eval" : loit_eval.mean(),
               "lore_losses" : lore_losses.mean(),
               "lore_eval" : lore_eval.mean()})
    if epoch % 10 == 0:
        save_ee_plot(next_carry[0], next_carry[0],
                     os.path.join(checkpoint_dir, f"{epoch}_reward"),
f"""exploration and exploitation reward for different contexts after 
{epoch*1000} training updates, with exploit loss {loit_losses.mean():.4f}
and explore loss {lore_losses.mean():.4f}""")
        coverage_plot(next_carry[0],
                      os.path.join(checkpoint_dir, f"{epoch}_coverage"),
f"""Coverage of different policies
{epoch*1000} training updates, with exploit loss {loit_losses.mean():.4f}
and explore loss {lore_losses.mean():.4f}""")
        save("run_data", next_carry, epoch) # very rudimentary checkpointing
    epoch += 1

save_ee_plot(next_carry[0], next_carry[0],
             os.path.join(checkpoint_dir, f"{epoch}_reward"),
f"""exploration and exploitation reward for different contexts after 
{epoch*1000} training updates, with exploit loss {loit_losses.mean():.4f}
and explore loss {lore_losses.mean():.4f}""")
coverage_plot(next_carry[0],
              os.path.join(checkpoint_dir, f"{epoch}_coverage"),
f"""Coverage of different policies
{epoch*1000} training updates, with exploit loss {loit_losses.mean():.4f}
and explore loss {lore_losses.mean():.4f}""")
save("run_data", next_carry, epoch) # very rudimentary checkpointing