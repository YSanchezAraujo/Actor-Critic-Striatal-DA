import sys
import numpy as np
from jax.scipy.special import expit
import jax.numpy as jnp
from jax import lax, jit, random
from functools import partial
from jax.nn import one_hot
sys.path.append("/Users/ysa/Desktop/projects/actor_critic_models")

from three_param_aux_funcs_jax import (
    generate_fuzzy_x
)

def value_estimate(x, w): return jnp.dot(x, w)

def reward_function(next_loc, reward_loc, reward, no_reward):
    return lax.cond(
        next_loc == reward_loc,
        lambda _ : reward,
        lambda _ : no_reward,
        operand = None
    )

def get_next_state(loc, action):
    return lax.cond(
        action == False,
        lambda x: -1 + x,
        lambda x: 1 + x,
        loc
    )

def get_action_sign(action):
    return lax.cond(
        action == False,
        lambda _ : -1,
        lambda _ : 1,
        operand = None
    )

def get_action_index(action):
    return lax.cond(
        action == False,
        lambda _ : 0,
        lambda _ : 1,
        operand = None
    )

def initialize_parameters(bonus_vals, actor_inits):
    w = jnp.array(bonus_vals)
    theta = jnp.array(actor_inits)
    return theta, w

def ascent_step_w(w, x_loc, delta, lr_w):
    return w + lr_w * delta * x_loc

def ascent_step_theta(theta, x_loc, delta, lr_theta, action, p_right):
    jac_theta = x_loc * p_right * (1 - p_right) * get_action_sign(action)
    return theta + lr_theta * delta * jac_theta, jac_theta

def ascent_step_logtheta(theta, x_loc, delta, lr_theta, p_right, action):
    y = get_action_index(action)
    jac_log_theta = x_loc * (y - p_right)
    return theta + lr_theta * delta * jac_log_theta, jac_log_theta

def scale_state(x_loc, loc, current_side):
    proximity_map = jnp.array([
        0,    # 0: terminal incorrect 1
        0.1,  # 1: delay 1
        0.3,  # 2: left start
        0.5,  # 3: delay 2
        0,    # 4: terminal correct
        0.5,  # 5: delay 3
        0.3,  # 6: right Start
        0.1,  # 7: delay 4
        0     # 8: terminal incorrect 2
    ])
        
    one_hot_vec = one_hot(current_side+1, num_classes=3) * proximity_map[loc]

    return x_loc + one_hot_vec



def simulate_trial(carry, loop_values, params):
    # unpack everything
    stim, side = loop_values

    (
        theta, 
        w, 
        jac_theta_traj, 
        v_pos_traj, 
        v_nextpos_traj, 
        delta_traj, 
        delta_delay_traj, 
        p_right_traj,
        action_traj,
        reward_traj,
        key,
        counter
     ) = carry
    
    loc = params.start_locs[side]

    key, subkey = random.split(key)

    x_loc, key = generate_fuzzy_x(
        key,
        params.convals[stim],   # stim val
        params.reward_loc,
        params.incorrect_locs,
        loc, # loc
        params.bias_val_value, # bias val
        params.obs_scale, # obs scale
        side,
    )

    x_loc_scaled = scale_state(x_loc, loc, side)
    v_pos = value_estimate(x_loc_scaled, w)
    v_pos_traj = v_pos_traj.at[counter].set(v_pos)
    p_right = expit(jnp.dot(x_loc_scaled, theta))
    p_right_traj = p_right_traj.at[counter].set(p_right)
    key, subkey = random.split(key)
    action = random.bernoulli(subkey, p=p_right)
    action_traj = action_traj.at[counter].set(action)

    next_loc = get_next_state(loc, action)

    x_nextloc, key = generate_fuzzy_x(
        key,
        params.convals[stim],   # stim val
        params.reward_loc,
        params.incorrect_locs,
        next_loc, # loc
        params.bias_val_value, # bias val
        params.obs_scale, # obs scale
        side,
    )

    x_nextloc_scaled = scale_state(x_nextloc, next_loc, side)

    key, subkey = random.split(key)

    v_nextpos = value_estimate(x_nextloc_scaled, w)
    v_nextpos_traj = v_nextpos_traj.at[counter].set(v_nextpos)
    delta_delay = jnp.maximum(0.0, params.gamma * v_nextpos - v_pos)
    delta_delay_traj = delta_delay_traj.at[counter].set(delta_delay)
    theta, jac_theta = ascent_step_logtheta(theta, x_loc_scaled, delta_delay, params.lr_theta, p_right, action)
    jac_theta_traj = jac_theta_traj.at[counter].set(jac_theta)

    loc = next_loc
    next_loc = get_next_state(loc, action)

    R_t = reward_function(next_loc, params.reward_loc, params.reward_val, params.noreward_val)
    delta = R_t - v_nextpos
    delta_traj = delta_traj.at[counter].set(delta)
    reward_traj = reward_traj.at[counter].set(R_t)

    w = ascent_step_w(w, x_nextloc_scaled, delta, params.lr_w)

    counter += 1
    learnable_weights = (theta, w)

    new_carry = (
        theta, 
        w, 
        jac_theta_traj,
        v_pos_traj, 
        v_nextpos_traj, 
        delta_traj, 
        delta_delay_traj, 
        p_right_traj,
        action_traj,
        reward_traj,
        key,
        counter
        )

    return new_carry, learnable_weights

@jit
def run_agent(key, params, loop_values):
    theta, w = initialize_parameters(params.bonus_vals, params.actor_inits)

    n = loop_values[0].shape[0]
    p = theta.shape[0]

    init_carry = (
        theta, 
        w, 
        jnp.zeros((n, p)),          # jac_theta_traj
        jnp.zeros((n, )),           # v_pos_traj
        jnp.zeros((n, )),           # v_nextpos_traj
        jnp.zeros((n, )),           # delta_traj
        jnp.zeros((n, )),           # delta_delay_traj
        jnp.zeros((n, )),           # p_right_traj
        jnp.zeros((n, ), dtype=bool),  # action_traj
        jnp.zeros((n, )),           # reward_traj
        key,
        0                            # counter
    )

    f_partial = partial(simulate_trial, params=params)
    final_carry, weights = lax.scan(f_partial, init_carry, loop_values)
    return final_carry, weights


"""
this is the inverted model
""";


def simulate_trial_inverted(carry, loop_values, params):
    # unpack everything
    stim, side = loop_values

    (
        theta, 
        w, 
        jac_theta_traj, 
        v_pos_traj, 
        v_nextpos_traj, 
        delta_traj, 
        delta_delay_traj, 
        p_right_traj,
        action_traj,
        reward_traj,
        key,
        counter
     ) = carry
    
    # get initial location
    loc = params.start_locs[side]
    key, subkey = random.split(key)

    x_loc, key = generate_fuzzy_x(
        key,
        params.convals[stim],   # stim val
        params.incorrect_locs[side],
        params.terminal_states,
        loc, # loc
        params.bias_val_value, # bias val
        params.obs_scale, # obs scale
        side,
    )

    x_loc_scaled = scale_state(x_loc, loc, side)

    v_pos = value_estimate(x_loc_scaled, w)
    v_pos_traj = v_pos_traj.at[counter].set(v_pos)
    p_right = expit(jnp.dot(x_loc_scaled, theta))
    p_right_traj = p_right_traj.at[counter].set(p_right)
    key, subkey = random.split(key)
    action = random.bernoulli(subkey, p=p_right)
    action_traj = action_traj.at[counter].set(action)

    next_loc = get_next_state(loc, action)

    key, subkey = random.split(key)

    x_nextloc, key = generate_fuzzy_x(
        key,
        params.convals[stim],   # stim val
        params.incorrect_locs[side],
        params.terminal_states,
        next_loc, # loc
        params.bias_val_value, # bias val
        params.obs_scale, # obs scale
        side,
    )

    x_nextloc_scaled = scale_state(x_nextloc, next_loc, side)

    key, subkey = random.split(key)

    v_nextpos = value_estimate(x_nextloc_scaled, w)
    v_nextpos_traj = v_nextpos_traj.at[counter].set(v_nextpos)
    delta_delay = jnp.maximum(0.0, params.gamma * v_nextpos - v_pos)

    delta_delay_traj = delta_delay_traj.at[counter].set(delta_delay)
    theta, jac_theta = ascent_step_logtheta(theta, x_loc_scaled, delta_delay, params.lr_theta, p_right, action)
    jac_theta_traj = jac_theta_traj.at[counter].set(jac_theta)

    loc = next_loc
    next_loc = get_next_state(loc, action)

    R_t = reward_function(next_loc, params.reward_loc[side], params.reward_val, params.noreward_val)
    delta = R_t - v_nextpos
    delta_traj = delta_traj.at[counter].set(delta)
    reward_traj = reward_traj.at[counter].set(R_t)

    w = ascent_step_w(w, x_nextloc_scaled, delta, params.lr_w)

    counter += 1
    learnable_weights = (theta, w)

    new_carry = (
        theta, 
        w, 
        jac_theta_traj,
        v_pos_traj, 
        v_nextpos_traj, 
        delta_traj, 
        delta_delay_traj, 
        p_right_traj,
        action_traj,
        reward_traj,
        key,
        counter
        )

    return new_carry, learnable_weights

@jit
def run_agent_inverted(key, params, loop_values):
    theta, w = initialize_parameters(params.bonus_vals, params.actor_inits)

    n = loop_values[0].shape[0]
    p = theta.shape[0]

    init_carry = (
        theta, 
        w, 
        jnp.zeros((n, p)),          # jac_theta_traj
        jnp.zeros((n, )),           # v_pos_traj
        jnp.zeros((n, )),           # v_nextpos_traj
        jnp.zeros((n, )),           # delta_traj
        jnp.zeros((n, )),           # delta_delay_traj
        jnp.zeros((n, )),           # p_right_traj
        jnp.zeros((n, ), dtype=bool),  # action_traj
        jnp.zeros((n, )),           # reward_traj
        key,                          # random key
        0                            # counter
    )

    f_partial = partial(simulate_trial_inverted, params=params)
    final_carry, weights = lax.scan(f_partial, init_carry, loop_values)
    return final_carry, weights
