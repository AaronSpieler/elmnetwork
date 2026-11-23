import copy
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def cast_floats(pytree: Any, target_dtype_str: str) -> Any:
    target_dtype = jnp.dtype(target_dtype_str)

    def _cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(target_dtype)
        return x

    return jtu.tree_map(_cast, pytree)


def cosine_decay(curr_step, decay_steps, min_val=0.0, max_val=1.0):
    if curr_step < 0:
        return max_val
    if decay_steps == 0:
        return min_val
    if curr_step >= decay_steps:
        return min_val
    progress = curr_step / decay_steps
    return min_val + 1 / 2 * (max_val - min_val) * (1 + math.cos(math.pi * progress))


def copy_model_to_cpu(model: eqx.Module):
    """Note: should not be used to store copies for long periods of time"""
    copy_to_cpu_fn = lambda x: jax.device_put(jnp.copy(x), jax.devices("cpu")[0])
    return jax.tree_util.tree_map(
        lambda x: copy_to_cpu_fn(x) if eqx.is_array(x) else x, model
    )


def _tree_flatten_arrays_with_paths(tree, prefix="", apply_fun=None):
    if isinstance(tree, eqx.Module):
        for name, value in tree.__dict__.items():
            yield from _tree_flatten_arrays_with_paths(
                value, prefix + name + ".", apply_fun
            )
    elif isinstance(tree, (list, tuple)):
        for idx, value in enumerate(tree):
            yield from _tree_flatten_arrays_with_paths(
                value, prefix + f"[{idx}].", apply_fun
            )
    elif isinstance(tree, dict):
        for key, value in tree.items():
            yield from _tree_flatten_arrays_with_paths(
                value, prefix + f"{key}.", apply_fun
            )
    elif eqx.is_array(tree):
        yield {prefix[:-1]: tree if apply_fun is None else apply_fun(tree)}
    else:
        pass


def tree_flatten_arrays_with_paths(tree, apply_fun=None):
    list_of_dicts = _tree_flatten_arrays_with_paths(tree, apply_fun=apply_fun)
    merged_dict = {}
    for d in list_of_dicts:
        merged_dict.update(d)
    return merged_dict


def calculate_model_cost_and_weight_stats(
    model,
    train_params_filter,
    data_shape,
    comp_key,
    data_dtype=jnp.float32,
):
    filtered_model = eqx.filter(model, train_params_filter)
    model_stats = {}

    # compute cost statistics (unfused view for fused models so XLA reports FLOPs)
    cost_model = model
    if getattr(model, "fused", False):
        cost_model = copy.copy(model)
        object.__setattr__(cost_model, "fused", False)
    one_step_data = jnp.ones(data_shape, dtype=data_dtype)
    cost_fun = lambda key, x: cost_model(
        key=key,
        x=x,
    )
    cost_analysis = (
        jax.jit(cost_fun).lower(comp_key, one_step_data).compile().cost_analysis()
    )
    try:  # old formatting
        model_stats["cost_analysis"] = cost_analysis[0]
    except:
        model_stats["cost_analysis"] = cost_analysis

    # parameter statistics
    model_stats["params"] = tree_flatten_arrays_with_paths(model, lambda x: x.size)
    model_stats["total_params"] = sum(model_stats["params"].values())
    model_stats["trainable_params"] = tree_flatten_arrays_with_paths(
        filtered_model, lambda x: x.size
    )
    model_stats["total_trainable_params"] = sum(
        model_stats["trainable_params"].values()
    )

    return model_stats


def merge_list_of_dicts(
    list_of_dicts: list[dict[str, Any]], stack_fun=None
) -> tuple[list[str], list[Any]]:

    merged_dict: dict[str, list[Any]] = {}
    for curr_dict in list_of_dicts:
        for param_name, value in curr_dict.items():
            if param_name not in merged_dict:
                merged_dict[param_name] = []
            merged_dict[param_name].append(value)

    dict_keys = list(merged_dict.keys())
    dict_vals = [
        (
            merged_dict[param_name]
            if stack_fun is None
            else stack_fun(merged_dict[param_name])
        )
        for param_name in dict_keys
    ]

    return dict_keys, dict_vals
