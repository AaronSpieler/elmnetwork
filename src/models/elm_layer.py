import copy
import warnings
from typing import Any, Optional

import equinox as eqx
import jax.lax as jlx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Float, Integer, PRNGKeyArray, Shaped, jaxtyped
from typeguard import typechecked as typechecker

from .abstract_model import DynamicModel
from .elm_modeling_utils import random_connect, random_connect_pooled
from .elm_neuron import (
    ELM,
    NEURON_MONITORS,
    Input,
    Monitor,
    NeuronCarry,
    NeuronRecord,
)

# below properties are vectorized neuron properties
EnsembleState = dict[str, Float[Array, "neuron ..."]]
EnsembleCarry = tuple[
    dict[str, Float[Array, "neuron ..."]],
    Shaped[PRNGKeyArray, "neuron"],
    Shaped[PRNGKeyArray, "neuron"],
]
EnsembleRecord = dict[str, Float[Array, "neuron ..."]]
EnsembleRecordHistory = dict[str, Float[Array, "time neuron ..."]]

LayerActivity = Float[Array, "neuron"]
LayerActHistory = Float[Array, "time neuron"]
LayerState = dict[str, Float[Array, "..."]]
LayerCarry = tuple[LayerState, PRNGKeyArray, PRNGKeyArray]
LayerRecord = dict[str, Float[Array, "..."]]
LayerRecordHistory = dict[str, Float[Array, "time ..."]]

LayerEnsembleState = tuple[LayerState, EnsembleState]
LayerEnsembleCarry = tuple[LayerCarry, EnsembleCarry]
LayerEnsembleRecord = tuple[LayerRecord, EnsembleRecord]
LayerEnsembleRecordHistory = tuple[LayerRecordHistory, EnsembleRecordHistory]
LayerMonitor = tuple[Monitor, Monitor]

# neuron with approx. 750 + 2500 = 3250 params
DEFAULT_ELM_LAYER_NEURON_CONFIG = {
    "num_memory": 10,
    "high_pass_tau": 5.0,
    "neuron_activation": "relu",
}

LAYER_MONITORS = [
    "activity",
]
ALL_MONITORS = (LAYER_MONITORS, NEURON_MONITORS)
INPUT_WIRINGS = [
    "random",
    "random_pooled",
]


# NOTE:
# accessing ensemble.method(...) or ensemble.property directly
# without vmap wrapper can result in errors or incorrect values


@jaxtyped(typechecker=typechecker)
class ELMLayer(eqx.Module, DynamicModel):
    """A layer of Expressive Leaky Memory (ELM) neurons.
    Default neuron has d_m = 10, high-pass filter with tau_r = 5.0, and ReLU activation.
    Unless manually specified neuron d_m = 2*d_mlp, d_tree = 2*d_mlp and d_branch = d_tree.
    """

    num_input: int = eqx.field(static=True)
    num_neuron: int = eqx.field(static=True)

    # wiring configuration:
    input_wiring: str = eqx.field(static=True)
    input_wiring_args: dict = eqx.field(static=True)

    # general setup:
    input_dropout: float = eqx.field(static=True)
    recurrent_dropout: float = eqx.field(static=True)
    fused_kernels_args: tuple = eqx.field(static=True)

    # learnable:
    ensemble: ELM

    # not learnable:
    input_connectome: Integer[Array, "neuron synapse"]
    input_pool_order: Optional[Integer[Array, "branch neuron"]]

    def __init__(
        self,
        num_input: int,
        num_neuron: int = 1024,
        neuron_config: Optional[dict] = None,
        input_wiring: str = "random",
        input_wiring_args: Optional[dict] = None,
        input_dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        fused_kernels_args: Optional[dict] = None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `num_input`: The number of input channels to the layer.
        - `num_neuron`: The number of neurons in the layer.
        - `neuron_config`: The configuration for the individual ELM neuron. Its input
            and output size is auto-configured, and no custom synapse routing is allowed.
        - `input_wiring`: Which strategy to use for connecting inputs to neuron synapses.
            The default is `random` which uniformly at random samples possible connections.
            `random_pooled` samples alike, but groups of `share_size` neurons share each
            branch's input sources (branch pools; see `random_connect_pooled`).
        - `input_wiring_args`: The arguments to the connectivity generating function.
            For the `random` wiring connectivity one can specify the probability of
            sampling a recurrent connection (instead of a feed-forward connection).
        - `input_dropout`: The dropout probability for the feed-forward input during training.
        - `recurrent_dropout`: The dropout probability for the recurrent input during training.
        - `fused_kernels_args`: Optional settings for the fused Triton kernels when the
            network runs with `fused=True` (see `fused_layer.FUSED_SETTINGS`).
        """
        super().__init__()

        self.num_input = num_input
        self.num_neuron = num_neuron
        self.input_wiring = input_wiring
        self.input_wiring_args = {} if input_wiring_args is None else input_wiring_args
        self.input_dropout = input_dropout
        self.recurrent_dropout = recurrent_dropout
        self.fused_kernels_args = tuple(sorted((fused_kernels_args or {}).items()))

        input_connectome_key, _, ensemble_key, key = jrandom.split(key, 4)

        # basic sanity checks
        assert neuron_config is None or "num_input" not in neuron_config
        assert neuron_config is None or "num_output" not in neuron_config
        assert neuron_config is None or "input_to_synapse_routing" not in neuron_config
        assert self.num_input > 0
        assert self.num_neuron > 0
        assert self.input_wiring in INPUT_WIRINGS
        assert 0 <= self.input_dropout < 1
        assert 0 <= self.recurrent_dropout < 1

        # finalizing neuron configuration
        og_neuron_config = copy.deepcopy(neuron_config) if neuron_config else {}
        neuron_config = copy.deepcopy(DEFAULT_ELM_LAYER_NEURON_CONFIG)
        neuron_config.update(og_neuron_config)
        assert isinstance(neuron_config["num_memory"], int)
        if neuron_config.get("mlp_num_hidden_units", None) is None:
            neuron_config["mlp_num_hidden_units"] = 2 * neuron_config["num_memory"]
            warnings.warn(
                f"Auto-configured neuron mlp_num_hidden_units = "
                f"{neuron_config['mlp_num_hidden_units']}"
            )
        if neuron_config.get("num_branch", None) is None:
            mlp_num_hidden_layer = neuron_config.get("mlp_num_hidden_layer", 1)
            if mlp_num_hidden_layer > 0:
                neuron_config["num_branch"] = 2 * neuron_config["mlp_num_hidden_units"]
            else:
                neuron_config["num_branch"] = 4 * neuron_config["num_memory"]
            warnings.warn(
                f"Auto-configured neuron num_branch = " f"{neuron_config['num_branch']}"
            )
        if neuron_config.get("num_synapse_per_branch", None) is None:
            neuron_config["num_synapse_per_branch"] = neuron_config["num_branch"]
            warnings.warn(
                f"Auto-configured neuron num_synapse_per_branch = "
                f"{neuron_config['num_synapse_per_branch']}"
            )
        neuron_config["num_input"] = int(
            neuron_config["num_branch"] * neuron_config["num_synapse_per_branch"]
        )

        # construct the input connectome matrix
        self.input_pool_order = None
        if self.input_wiring == "random":
            self.input_connectome = random_connect(
                key=input_connectome_key,
                num_input=self.num_input,
                num_neuron=self.num_neuron,
                num_neuron_inputs=neuron_config["num_input"],
                **self.input_wiring_args,
            )
        elif self.input_wiring == "random_pooled":
            self.input_connectome, self.input_pool_order = random_connect_pooled(
                key=input_connectome_key,
                num_input=self.num_input,
                num_neuron=self.num_neuron,
                num_branch_neuron=neuron_config["num_branch"],
                num_synapse_per_branch_neuron=neuron_config["num_synapse_per_branch"],
                return_order=True,
                **self.input_wiring_args,
            )
        else:
            raise NotImplementedError

        # construct the ensemble of neurons
        ensemble = self.construct_ensemble(
            ensemble_key,
            neuron_config,
        )
        self.ensemble = ensemble

    @property
    @jaxtyped(typechecker=typechecker)
    def ensemble_tau_m(self) -> Float[Array, "neuron timescales"]:
        return eqx.filter_vmap(lambda neuron: neuron.tau_m)(self.ensemble)

    @jaxtyped(typechecker=typechecker)
    def construct_ensemble(
        self,
        key: PRNGKeyArray,
        neuron_config: dict,
    ) -> ELM:
        keys = jrandom.split(key, self.num_neuron)
        return eqx.filter_vmap(lambda key: ELM(**neuron_config, num_output=1, key=key))(
            keys
        )

    @property
    def use_shared_lists(self) -> bool:
        return (
            self.input_wiring == "random_pooled"
            and self.input_pool_order is not None
            and self.ensemble.input_dropout == 0.0
        )

    @jaxtyped(typechecker=typechecker)
    def get_layer_init_state(self) -> LayerState:
        layer_state = {
            "a_t": jnp.zeros(self.num_neuron, dtype=self.ensemble.dtype),
        }
        return layer_state

    @jaxtyped(typechecker=typechecker)
    def get_ensemble_init_state(self) -> EnsembleState:
        return eqx.filter_vmap(lambda neuron: neuron.get_init_state())(self.ensemble)

    @jaxtyped(typechecker=typechecker)
    def get_init_state(self) -> LayerEnsembleState:
        return self.get_layer_init_state(), self.get_ensemble_init_state()

    @jaxtyped(typechecker=typechecker)
    def get_layer_init_carry(self, key: PRNGKeyArray) -> LayerCarry:
        dynamic_key, const_key = jrandom.split(key, 2)
        layer_state = self.get_layer_init_state()
        carry = (layer_state, dynamic_key, const_key)
        return carry

    @jaxtyped(typechecker=typechecker)
    def get_ensemble_init_carry(self, key: PRNGKeyArray) -> EnsembleCarry:
        return eqx.filter_vmap(lambda neuron, key: neuron.get_init_carry(key))(
            self.ensemble, jrandom.split(key, self.num_neuron)
        )

    @jaxtyped(typechecker=typechecker)
    def get_init_carry(self, key: PRNGKeyArray) -> LayerEnsembleCarry:
        layer_key, ensemble_key = jrandom.split(key, num=2)
        layer_init_carry = self.get_layer_init_carry(layer_key)
        ensemble_init_carry = self.get_ensemble_init_carry(ensemble_key)
        return layer_init_carry, ensemble_init_carry

    @eqx.filter_vmap(
        in_axes=dict(self=None, monitor=None, precomputed_branches=None)
    )
    def ensemble_dynamics(
        self,
        neuron: ELM,  # vectorized
        carry: NeuronCarry,  # vectorized
        x_t: Float[Array, "_"],  # vectorized
        monitor: list[str],  # broadcasted
        inference: bool,  # broadcasted
        precomputed_branches: bool = False,  # broadcasted
    ) -> tuple[NeuronCarry, NeuronRecord]:
        return neuron.dynamics(
            carry=carry,
            x_t=x_t,
            monitor=monitor,
            inference=inference,
            precomputed_branches=precomputed_branches,
        )

    @jaxtyped(typechecker=typechecker)
    def apply_activity_dropout(
        self,
        activity: Float[Array, "size"],
        dropout_probability: float,
        inference: bool,
        dropout_key: PRNGKeyArray,
    ) -> Float[Array, "size"]:
        if dropout_probability > 0.0 and not inference:
            dropout_mask = jrandom.bernoulli(
                key=dropout_key,
                p=dropout_probability,
                shape=activity.shape,
            )
            # scaling preserves mean
            activity = jnp.where(dropout_mask, 0, activity) / (1 - dropout_probability)
        return activity

    @jaxtyped(typechecker=typechecker)
    def gather_ensemble_inputs(
        self,
        layer_inputs: Float[Array, "channel"],
        prev_activity: Float[Array, "{self.num_neuron}"],
    ) -> Float[Array, "{self.num_neuron} ..."]:
        all_inputs = jnp.concatenate((layer_inputs, prev_activity), axis=-1)
        ensemble_inputs = jnp.take(all_inputs, indices=self.input_connectome, axis=-1)
        return ensemble_inputs

    @property
    @jaxtyped(typechecker=typechecker)
    def shared_lists(self) -> Optional[dict[str, Shaped[Array, "..."]]]:
        """Branch pools as shared lists, built once per sequence before scan."""
        if not self.use_shared_lists:
            return None
        n, nb = self.num_neuron, self.ensemble.num_branch
        p = self.ensemble.num_synapse_per_branch
        g = self.input_wiring_args.get("share_size", 16)
        runs = -(-n // g)  # the last list has fewer members unless n divides
        order = jnp.pad(
            self.input_pool_order, ((0, 0), (0, runs * g - n)), constant_values=-1
        )
        members = order.reshape(nb, runs, g)  # (branch, list, member), -1: no member
        real = members >= 0
        members = jnp.where(real, members, 0)
        branch = jnp.arange(nb)
        list_src = self.input_connectome.reshape(n, nb, p)[
            members[..., 0], branch[:, None]
        ]
        w_s = eqx.filter_vmap(lambda neuron: neuron.w_s)(self.ensemble)
        list_w = w_s.reshape(n, nb, p)[members, branch[:, None, None]]
        list_w = jnp.where(real[..., None], list_w, 0.0) * self.ensemble.input_scale
        return {
            "list_src": list_src,
            "list_w": list_w,
            "list_pos": jnp.argsort(self.input_pool_order, axis=1),
        }

    @jaxtyped(typechecker=typechecker)
    def sum_shared_lists(
        self,
        layer_inputs: Float[Array, "channel"],
        prev_activity: Float[Array, "{self.num_neuron}"],
        shared_lists: Optional[dict[str, Shaped[Array, "..."]]] = None,
    ) -> Float[Array, "{self.num_neuron} branch"]:
        tables = shared_lists if shared_lists is not None else self.shared_lists
        all_inputs = jnp.concatenate((layer_inputs, prev_activity), axis=-1)
        xs = all_inputs[tables["list_src"]]
        sums = jnp.einsum("krgp,krp->krg", tables["list_w"], xs)
        sums = jnp.take_along_axis(
            sums.reshape(sums.shape[0], -1), tables["list_pos"], axis=1
        )
        return sums.T

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def dynamics(
        self,
        carry: LayerEnsembleCarry,
        x_t: Float[Array, "channel"],
        monitor: Optional[LayerMonitor],
        inference: bool,
        shared_lists: Optional[dict[str, Shaped[Array, "..."]]] = None,
    ) -> tuple[LayerEnsembleCarry, LayerEnsembleRecord]:
        layer_carry, ensemble_carry = carry
        layer_monitor, ensemble_monitor = monitor if monitor else ([], [])
        prev_state, dynamic_key, const_key = layer_carry
        next_state: LayerState = {}

        # const key for variational dropout
        input_dropout_key, recurrent_dropout_key = jrandom.split(const_key, 2)
        x_t = self.apply_activity_dropout(
            x_t, self.input_dropout, inference, input_dropout_key
        )
        a_t_dropped = self.apply_activity_dropout(
            prev_state["a_t"], self.recurrent_dropout, inference, recurrent_dropout_key
        )

        # simulate the ensemble for one time step
        if self.use_shared_lists:
            ensemble_inputs = self.sum_shared_lists(x_t, a_t_dropped, shared_lists)
            precomputed_branches = True
        else:
            ensemble_inputs = self.gather_ensemble_inputs(x_t, a_t_dropped)
            precomputed_branches = False
        ensemble_carry, ensemble_recording = self.ensemble_dynamics(
            self.ensemble,
            ensemble_carry,
            ensemble_inputs,
            ensemble_monitor,
            inference,
            precomputed_branches,
        )
        next_state["a_t"] = jnp.squeeze(ensemble_recording["output"], axis=-1)

        # prepare carry and recordings
        if "output" not in ensemble_monitor:
            del ensemble_recording["output"]
        layer_recording = {"activity": next_state["a_t"]}
        layer_carry = (next_state, dynamic_key, const_key)
        carry = layer_carry, ensemble_carry
        recordings = layer_recording, ensemble_recording
        return carry, recordings

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        x: Input,
        monitor: Optional[LayerMonitor] = None,
        inference: bool = False,
        init_carry: Optional[LayerEnsembleCarry] = None,
        *,
        key: PRNGKeyArray,
    ) -> tuple[LayerActHistory, LayerEnsembleRecordHistory, LayerEnsembleCarry]:
        # compute the recurrent dynamics for a single sample
        x = x.astype(self.ensemble.dtype)
        shared_lists = self.shared_lists
        scan_fun = lambda carry_state, input_spikes: self.dynamics(
            carry_state,
            input_spikes,
            monitor,
            inference,
            shared_lists,
        )
        carry = init_carry if init_carry is not None else self.get_init_carry(key=key)
        carry, records = jlx.scan(scan_fun, carry, x)
        return records[0]["activity"], records, carry

    # part of model for ease of use
    def get_train_params_filter(self) -> Any:
        elm_layer_filter = jtu.tree_map(lambda x: eqx.is_inexact_array(x), self)
        elm_layer_filter = eqx.tree_at(
            lambda tree: (
                tree.ensemble,
                tree.input_connectome,
                tree.input_pool_order,
            ),
            elm_layer_filter,
            replace=(
                # below works because neuron and ensemble share pytree structure
                self.ensemble.get_train_params_filter(),
                False,
                False,
            ),
            is_leaf=lambda x: x is None,
        )
        return elm_layer_filter
