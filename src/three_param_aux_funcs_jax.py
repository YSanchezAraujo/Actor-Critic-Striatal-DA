import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from jax import lax, vmap
from jax.nn import one_hot

eps_float32 = jnp.finfo(jnp.float32).eps
eps_float64 = jnp.finfo(jnp.float64).eps

def generate_data(
        key: random.PRNGKey, 
        n_trials: int
) -> tuple[jnp.ndarray, jnp.ndarray, random.PRNGKey]:
    """
    Generate data for the AC model.
    """
    contrast_labs = jnp.array([0, 1, 2, 3])
    side_labs = jnp.array([0, 1])
    key, subkey = random.split(key)
    sides = random.choice(subkey, side_labs, (n_trials, ))
    key, subkey = random.split(key)
    stims = random.choice(subkey, contrast_labs, (n_trials, ))
    return stims, sides, key

"""
#0S0X0S0#
"""
def generate_single_state_space(n_states_per_side: int) -> tuple[jnp.ndarray, dict]:
    """
    Generate a single state space for the AC model.
    """
    total_states = 5 + n_states_per_side * 2 
    state_space = jnp.zeros(total_states)
    TERMINAL_IL_VAL = -1.0
    TERMINAL_IR_VAL = 1.0
    TERMINAL_GOAL = 4.0
    START_R = 3.0
    START_L = 6.0
    state_space = state_space.at[0].set(TERMINAL_IL_VAL) 
    state_space = state_space.at[-1].set(TERMINAL_IR_VAL)
    state_space = state_space.at[2 + n_states_per_side].set(TERMINAL_GOAL)
    state_space = state_space.at[2].set(START_L)
    state_space = state_space.at[len(state_space)-3].set(START_R)

    loc_info = {
        "reward":jnp.where(state_space == TERMINAL_GOAL)[0][0],
	    "incorrect_left":jnp.where(state_space == TERMINAL_IL_VAL)[0][0],
	    "incorrect_right":jnp.where(state_space == TERMINAL_IR_VAL)[0][0],
	    "start_left":jnp.where(state_space == START_L)[0][0],
	    "start_right":jnp.where(state_space == START_R)[0][0]
    }

    state_space = jnp.zeros(total_states)

    return state_space, loc_info

def generate_single_state_space_inverted(n_states_per_side: int) -> tuple[jnp.ndarray, dict]:
    """
    Generate a single state space for the ACI model.
    """
    total_states = 5 + n_states_per_side * 2 
    state_space = jnp.zeros(total_states)
    TERMINAL_CL_VAL = 4.0    # Terminal goal for the leftmost position
    TERMINAL_CR_VAL = -4.0    # Terminal goal for the rightmost position
    TERMINAL_INCORRECT = -1.0  # Incorrect state for turning towards the center
    START_R = 1.0  # Start state on the right
    START_L = 2.0  # Start state on the left
    
    #state_space[0] = TERMINAL_CL_VAL  # Leftmost position as the goal state
    state_space = state_space.at[0].set(TERMINAL_CL_VAL)
    #state_space[-1] = TERMINAL_CR_VAL  # Rightmost position as the goal state
    state_space = state_space.at[-1].set(TERMINAL_CR_VAL)
    #state_space[2 + n_states_per_side] = TERMINAL_INCORRECT  # Incorrect state in the center
    state_space = state_space.at[2 + n_states_per_side].set(TERMINAL_INCORRECT)
    #state_space[2] = START_L  # Start state on the left
    state_space = state_space.at[2].set(START_L)
    #state_space[len(state_space)-3] = START_R  # Start state on the right
    state_space = state_space.at[len(state_space)-3].set(START_R)

    loc_info = {
        "reward_left": jnp.where(state_space == TERMINAL_CL_VAL)[0][0],
        "reward_right": jnp.where(state_space == TERMINAL_CR_VAL)[0][0],
        "incorrect_left": jnp.where(state_space == TERMINAL_INCORRECT)[0][0],
        "incorrect_right": jnp.where(state_space == TERMINAL_INCORRECT)[0][0],
        "start_left": jnp.where(state_space == START_L)[0][0],
        "start_right": jnp.where(state_space == START_R)[0][0]
    }

    state_space = jnp.zeros(total_states)

    return state_space, loc_info    

def bonus_values(l_val: float, r_val: float) -> jnp.ndarray:
    x = jnp.zeros((3,))
    x = x.at[1].set(l_val)
    x = x.at[2].set(r_val)
    return x

def generate_fuzzy_x(
    key: random.PRNGKey,
    conval_stim: float,        
    reward_loc: int,
    incorrect_locs: jnp.ndarray,
    loc: int,
    bias_val: float,
    obs_scale: float,
    current_side: int,
) -> tuple[random.PRNGKey, jnp.ndarray]:
    
    key, subkey = random.split(key)

    is_reward = (loc == reward_loc)
    is_incorrect = jnp.any(loc == incorrect_locs)
    is_terminal = jnp.logical_or(is_reward, is_incorrect)


    def terminal_case(_):
        x = jnp.zeros(3)
        return x

    def non_terminal_case(operand):
        key = operand

        x_bias = one_hot(0, num_classes=3) # * bias_val # old version
        one_hot_vec = one_hot(current_side+1, num_classes=3) * conval_stim 
        # loc_scale_vec = one_hot(current_side+1, num_classes=3) 
        # scaled_one_hot = one_hot_vec + loc_scale_vec

        x = x_bias + one_hot_vec
        noise = random.normal(key, shape=(3,)) * obs_scale
        x = x + noise
        x = x.at[0].set(bias_val)

        return x

    xpos = lax.cond(
        is_terminal,
        terminal_case,
        non_terminal_case,
        operand=subkey
    )

    return xpos, key
