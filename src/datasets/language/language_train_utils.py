from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import random as jrandom
from jaxtyping import Array, Float, Integer, PRNGKeyArray, jaxtyped
from typeguard import typechecked as typechecker

from ...models.abstract_model import DynamicModel
from ...training.regularizer import Regularizer

LANGUAGE_INPUT = Integer[Array, "batch time"]
LANGUAGE_TARGET = Integer[Array, "batch time"]
LANGUAGE_PRED = Float[Array, "batch time vocabsize"]


# NOTE: adding label smoothing or reularization changes loss,
#       therefore must be disabled for BPC calculation
@eqx.filter_jit(donate="all-except-first")
@jaxtyped(typechecker=typechecker)
def language_loss(
    model: DynamicModel,  # NOTE: memory will NOT be donated
    x: LANGUAGE_INPUT,
    y: LANGUAGE_TARGET,
    key: PRNGKeyArray,
    inference: bool,
    batch_size: int,
    vocab_size: int,
    burn_in_time: int = 0,
    label_smoothing: float = 0.0,
    init_carry: Optional[Any] = None,
    return_logits: bool = False,
    return_carry: bool = False,
    regularizer: Optional[Regularizer] = None,
    grad_checkpointing: bool = False,
) -> Tuple[Optional[Float[Array, ""]], Tuple[Optional[LANGUAGE_PRED], Optional[Any]]]:
    # logits: batch, time, classes
    keys = jrandom.split(key, batch_size)
    logits, recording, carry = jax.vmap(
        lambda x, init_carry, key: model(
            x=x,
            init_carry=init_carry,
            key=key,
            monitor=regularizer.monitor if regularizer else None,
            inference=inference,
            grad_checkpointing=grad_checkpointing,
        )
    )(x, init_carry, keys)
    full_logits = logits
    if burn_in_time > 0:
        logits = logits[:, burn_in_time:]
        y = y[:, burn_in_time:]
    y = jax.nn.one_hot(y, vocab_size)
    if label_smoothing > 0:
        y = optax.smooth_labels(y, label_smoothing)
    loss = jnp.mean(optax.safe_softmax_cross_entropy(logits, y))
    if regularizer is not None:
        loss = loss + jnp.mean(eqx.filter_vmap(lambda x: regularizer(x))(recording))
    logits = full_logits if return_logits else None
    carry = carry if return_carry else None
    return loss, (logits, carry)


@eqx.filter_value_and_grad(has_aux=True)
@jaxtyped(typechecker=typechecker)
def language_loss_and_grads(
    model_diff: DynamicModel,
    model_stat: DynamicModel,
    x: LANGUAGE_INPUT,
    y: LANGUAGE_TARGET,
    key: PRNGKeyArray,
    batch_size: int,
    vocab_size: int,
    burn_in_time: int,
    label_smoothing: float,
    init_carry: Optional[Any] = None,
    return_logits: bool = False,
    return_carry: bool = False,
    regularizer: Optional[Regularizer] = None,
    grad_checkpointing: bool = False,
) -> Tuple[Optional[Float[Array, ""]], Tuple[Optional[LANGUAGE_PRED], Optional[Any]]]:
    model = eqx.combine(model_diff, model_stat)
    return language_loss(
        model,  # NOTE: memory will NOT not be donated
        x,
        y,
        key,
        False,
        batch_size,
        vocab_size,
        burn_in_time,
        label_smoothing,
        init_carry,
        return_logits,
        return_carry,
        regularizer,
        grad_checkpointing,
    )


@eqx.filter_jit(donate="all")
def language_make_step(
    model: DynamicModel,
    x: LANGUAGE_INPUT,
    y: LANGUAGE_TARGET,
    opt_state: optax.OptState,
    key: PRNGKeyArray,
    batch_size: int,
    vocab_size: int,
    burn_in_time: int,
    label_smoothing: float,
    train_params_filter: DynamicModel,
    gradient_transform: optax.GradientTransformation,
    init_carry: Optional[Any] = None,
    return_logits: bool = False,
    return_grad_norms: bool = False,
    return_carry: bool = False,
    regularizer: Optional[Regularizer] = None,
    grad_checkpointing: bool = False,
) -> Tuple[
    Optional[Float[Array, ""]],
    Optional[LANGUAGE_PRED],
    Optional[DynamicModel],
    DynamicModel,
    Any,
    Optional[Any],
]:
    model_diff, model_stat = eqx.partition(model, train_params_filter)
    (loss, (logits, carry)), grads = language_loss_and_grads(
        model_diff,
        model_stat,
        x,
        y,
        key,
        batch_size,
        vocab_size,
        burn_in_time,
        label_smoothing,
        init_carry,
        return_logits,
        return_carry,
        regularizer,
        grad_checkpointing,
    )
    updates, opt_state = gradient_transform.update(
        grads, opt_state, eqx.filter(model, train_params_filter)
    )
    model = eqx.apply_updates(model, updates)
    if return_grad_norms:
        grad_norms = jtu.tree_map(
            lambda x: jnp.linalg.norm(x) if eqx.is_array(x) else x, grads
        )
    else:
        grad_norms = None
    return loss, logits, grad_norms, model, opt_state, carry
