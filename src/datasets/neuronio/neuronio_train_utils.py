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

NEURONIO_INPUT = Float[Array, "batch time channel"]
NEURONIO_TARGET = Tuple[Integer[Array, "batch time"], Float[Array, "batch time"]]


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def neuronio_loss(
    model: DynamicModel,
    x: NEURONIO_INPUT,
    y: NEURONIO_TARGET,
    key: PRNGKeyArray,
    batch_size: int,
    burn_in_time: int,
    inference: bool,
    init_carry: Optional[Any] = None,
    return_carry: bool = False,
) -> Tuple[Optional[Float[Array, ""]], Optional[Any]]:
    keys = jrandom.split(key, batch_size)
    # preds: [batch, time, 2]
    preds, _, carry = jax.vmap(
        lambda x, init_carry, key: model(
            x=x,
            init_carry=init_carry,
            key=key,
            inference=inference,
        )
    )(x, init_carry, keys)
    bce_pred = preds[:, burn_in_time:, 0]
    bce_target = y[0][:, burn_in_time:]
    mse_pred = preds[:, burn_in_time:, 1]
    mse_target = y[1][:, burn_in_time:]

    # Mean reduction across batch
    bce_loss = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(bce_pred, bce_target))
    mse_loss = jnp.mean(optax.losses.squared_error(mse_pred, mse_target))

    # Balance the losses with a factor of 0.5 each
    loss = 0.5 * bce_loss + 0.5 * mse_loss
    carry = carry if return_carry else None
    return loss, carry


@eqx.filter_value_and_grad(has_aux=True)
@jaxtyped(typechecker=typechecker)
def neuronio_loss_and_grads(
    model_diff: DynamicModel,
    model_stat: DynamicModel,
    x: NEURONIO_INPUT,
    y: NEURONIO_TARGET,
    key: PRNGKeyArray,
    batch_size: int,
    burn_in_time: int,
    init_carry: Optional[Any] = None,
    return_carry: bool = False,
) -> Tuple[Optional[Float[Array, ""]], Optional[Any]]:
    model = eqx.combine(model_diff, model_stat)
    return neuronio_loss(
        model, x, y, key, batch_size, burn_in_time, False, init_carry, return_carry
    )


@eqx.filter_jit
@jaxtyped(typechecker=typechecker)
def neuronio_make_step(
    model: DynamicModel,
    x: NEURONIO_INPUT,
    y: NEURONIO_TARGET,
    opt_state: Any,
    key: PRNGKeyArray,
    batch_size: int,
    train_params_filter: DynamicModel,
    gradient_transform: optax.GradientTransformation,
    burn_in_time: int,
    init_carry: Optional[Any] = None,
    return_grad_norms: bool = False,
    return_carry: bool = False,
) -> Tuple[
    Optional[Float[Array, ""]], Optional[DynamicModel], DynamicModel, Any, Optional[Any]
]:
    model_diff, model_stat = eqx.partition(model, train_params_filter)
    (loss, carry), grads = neuronio_loss_and_grads(
        model_diff,
        model_stat,
        x,
        y,
        key,
        batch_size,
        burn_in_time,
        init_carry,
        return_carry,
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
    return loss, grad_norms, model, opt_state, carry
