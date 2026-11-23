import copy
import math
import warnings
from typing import Optional, Tuple

import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

MACHINE_EPS = 1e-6


def scaled_sigmoid(x: Array, lower_bound: float, upper_bound: float) -> Array:
    return (upper_bound - lower_bound) * jnn.sigmoid(x) + lower_bound


def inverse_scaled_sigmoid(x: Array, lower_bound: float, upper_bound: float) -> Array:
    x = jnp.clip(x, lower_bound + MACHINE_EPS, upper_bound - MACHINE_EPS)
    return jnp.log((x - lower_bound) / (upper_bound - x))


def custom_tanh(x: Array):
    return jnp.tanh(x * 2 / 3) * 1.7159


def create_interlocking_indices(num_input: int) -> Array:
    half_num_input_data = num_input // 2
    half_range_steps = (jnp.arange(0, num_input) % 2) * half_num_input_data
    single_steps = jnp.arange(0, num_input) // 2
    return half_range_steps + single_steps


def create_overlapping_window_indices(
    num_input: int, num_windows: int, num_elements_per_window: int
) -> Tuple[Array, Array]:
    stride_size = math.ceil(num_input / num_windows)
    overlapping_indices = (
        jnp.expand_dims(jnp.arange(0, num_windows), axis=1) * stride_size
    ) + jnp.expand_dims(jnp.arange(0, num_elements_per_window), axis=0)
    valid_indices = (overlapping_indices < num_input).astype(overlapping_indices.dtype)
    overlapping_indices = jnp.clip(overlapping_indices, max=num_input - 1)
    return overlapping_indices.flatten(), valid_indices.flatten()


def random_connect(
    key: PRNGKeyArray,
    num_input: int,
    num_neuron: int,
    num_neuron_inputs: int,
    recurrent_connect: Optional[float] = None,
    ff_conn_probs: Optional[tuple] = None,
    rec_conn_probs: Optional[tuple] = None,
    **kwargs,
) -> Array:
    assert recurrent_connect is None or 0.0 <= recurrent_connect < 1.0
    assert ff_conn_probs is None or len(ff_conn_probs) == num_input
    assert rec_conn_probs is None or len(rec_conn_probs) == num_neuron

    if recurrent_connect is None:
        recurrent_connect = num_neuron / (num_input + num_neuron)
        warnings.warn(f"Auto-configured layer recurrent_connect = {recurrent_connect}")

    # connection probabilities
    ff_conn_probs = (
        jnp.ones(num_input) / num_input
        if ff_conn_probs is None
        else jnp.array(ff_conn_probs) / jnp.sum(jnp.array(ff_conn_probs))
    )
    rec_conn_probs = (
        jnp.ones(num_neuron) / num_neuron
        if rec_conn_probs is None
        else jnp.array(rec_conn_probs) / jnp.sum(jnp.array(rec_conn_probs))
    )
    connection_probabilities = jnp.concatenate(
        [
            ff_conn_probs * (1 - recurrent_connect),
            rec_conn_probs * recurrent_connect,
        ]
    )

    neuron_inputs_shape = (num_neuron, num_neuron_inputs)
    connectivity = jrandom.choice(
        key=key,
        a=num_input + num_neuron,
        p=connection_probabilities,
        shape=neuron_inputs_shape,
        replace=True,
    )
    return connectivity


def random_connect_pooled(
    key: PRNGKeyArray,
    num_input: int,
    num_neuron: int,
    num_branch_neuron: int,
    num_synapse_per_branch_neuron: int,
    share_size: int = 16,
    return_order: bool = False,
    **kwargs,
):
    """random_connect, but groups of share_size neurons share each branch's input
    sources (branch pools; each neuron keeps its own synapse weights).

    Per branch, the neurons are shuffled and each run of share_size neurons takes
    the branch's sources of the run's first neuron: every shared list has exactly
    share_size sharers (the last run fewer if num_neuron isn't a multiple), and
    the lists keep random_connect's statistics. share_size=1: no sharing.
    return_order: also return the order, (branch, position) -> neuron, from which
    the shared-list and fused paths rebuild the groups.
    """
    assert share_size >= 1
    key_conn, key_order = jrandom.split(key)
    num_neuron_inputs = num_branch_neuron * num_synapse_per_branch_neuron
    conn = random_connect(key_conn, num_input, num_neuron, num_neuron_inputs, **kwargs)
    conn = conn.reshape(num_neuron, num_branch_neuron, num_synapse_per_branch_neuron)
    # per branch: a random order of the neurons (position -> neuron), each neuron's
    # position in it, and the first neuron of its run of share_size
    order = jnp.argsort(
        jrandom.uniform(key_order, (num_branch_neuron, num_neuron)), axis=1
    )
    position = jnp.argsort(order, axis=1)
    first = jnp.take_along_axis(order, position // share_size * share_size, axis=1)
    conn = conn[first.T, jnp.arange(num_branch_neuron)].reshape(
        num_neuron, num_neuron_inputs
    )
    return (conn, order.astype(jnp.int32)) if return_order else conn


def _recursive_dict_merge(base_dict: dict, update_dict: dict) -> dict:
    # merely helper function that modifies base_dict in-place!
    for key, value in update_dict.items():
        if (
            key in base_dict
            and isinstance(base_dict[key], dict)
            and isinstance(value, dict)
        ):
            base_dict[key] = _recursive_dict_merge(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def recursive_dict_merge(base_dict: dict, update_dict: dict) -> dict:
    """Creates a new merged dict by recursively updating a copy of
    a base dictionary with values from an update dictionary."""
    new_dict = copy.deepcopy(base_dict)
    return _recursive_dict_merge(new_dict, update_dict)
