from typing import Any, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import random as jrandom
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray, jaxtyped
from typeguard import typechecked as typechecker

from ...models.abstract_model import DynamicModel
from ...training.regularizer import Regularizer

LAST_STEP_CLASS_INPUT = Bool[Array, "batch time channel"]
LAST_STEP_CLASS_TARGET = Integer[Array, "batch"]
LAST_STEP_CLASS_PRED = Float[Array, "batch classes"]


@eqx.filter_jit(donate="all-except-first")
@jaxtyped(typechecker=typechecker)
def shd_loss(
    model: DynamicModel,  # NOTE: memory will NOT be donated
    x: LAST_STEP_CLASS_INPUT,
    y: LAST_STEP_CLASS_TARGET,
    key: PRNGKeyArray,
    batch_size: int,
    num_classes: int,
    inference: bool,
    init_carry: Optional[Any] = None,
    return_logits: bool = False,
    return_carry: bool = False,
    label_smoothing: float = 0.0,
    regularizer: Optional[Regularizer] = None,
    grad_checkpointing: bool = False,
) -> Tuple[Float[Array, ""], Tuple[Optional[LAST_STEP_CLASS_PRED], Optional[Any]]]:
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
    logits = logits[:, -1, :]  # get last output
    y = jax.nn.one_hot(y, num_classes)
    if label_smoothing > 0:
        y = optax.smooth_labels(y, label_smoothing)
    loss = jnp.mean(optax.safe_softmax_cross_entropy(logits, y))
    if regularizer is not None:
        loss = loss + jnp.mean(eqx.filter_vmap(lambda x: regularizer(x))(recording))
    logits = logits if return_logits else None
    carry = carry if return_carry else None
    return loss, (logits, carry)


@eqx.filter_value_and_grad(has_aux=True)
@jaxtyped(typechecker=typechecker)
def shd_loss_and_grads(
    model_diff: DynamicModel,
    model_stat: DynamicModel,
    x: LAST_STEP_CLASS_INPUT,
    y: LAST_STEP_CLASS_TARGET,
    key: PRNGKeyArray,
    batch_size: int,
    num_classes: int,
    init_carry: Optional[Any] = None,
    return_logits: bool = False,
    return_carry: bool = False,
    label_smoothing: float = 0.0,
    regularizer: Optional[Regularizer] = None,
    grad_checkpointing: bool = False,
) -> Tuple[Float[Array, ""], Tuple[Optional[LAST_STEP_CLASS_PRED], Optional[Any]]]:
    model = eqx.combine(model_diff, model_stat)
    return shd_loss(
        model,  # NOTE: memory will NOT not be donated
        x,
        y,
        key,
        batch_size,
        num_classes,
        False,
        init_carry,
        return_logits,
        return_carry,
        label_smoothing,
        regularizer,
        grad_checkpointing,
    )


@eqx.filter_jit(donate="all")
@jaxtyped(typechecker=typechecker)
def shd_make_step(
    model: DynamicModel,
    x: LAST_STEP_CLASS_INPUT,
    y: LAST_STEP_CLASS_TARGET,
    opt_state: Any,
    key: PRNGKeyArray,
    batch_size: int,
    num_classes: int,
    train_params_filter: DynamicModel,
    gradient_transform: optax.GradientTransformation,
    init_carry: Optional[Any] = None,
    return_logits: bool = False,
    return_grad_norms: bool = False,
    return_carry: bool = False,
    label_smoothing: float = 0.0,
    regularizer: Optional[Regularizer] = None,
    grad_checkpointing: bool = False,
) -> Tuple[
    Optional[Float[Array, ""]],
    Optional[LAST_STEP_CLASS_PRED],
    Optional[DynamicModel],
    DynamicModel,
    Any,
    Optional[Any],
]:
    model_diff, model_stat = eqx.partition(model, train_params_filter)
    (loss, (logits, carry)), grads = shd_loss_and_grads(
        model_diff,
        model_stat,
        x,
        y,
        key,
        batch_size,
        num_classes,
        init_carry,
        return_logits,
        return_carry,
        label_smoothing,
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
