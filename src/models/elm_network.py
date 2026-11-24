import copy
import warnings
from typing import Any, Optional

import equinox as eqx
import jax
import jax.lax as jlx
import jax.nn as jnn
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Float, Integer, PRNGKeyArray, jaxtyped
from typeguard import typechecked as typechecker

from .abstract_model import DynamicModel
from .elm_layer import (
    DEFAULT_ELM_LAYER_NEURON_CONFIG,
    ELMLayer,
    Input,
    LayerActHistory,
    LayerEnsembleCarry,
    LayerEnsembleRecord,
    LayerEnsembleRecordHistory,
    LayerEnsembleState,
    LayerMonitor,
)
from .elm_modeling_utils import recursive_dict_merge

# NOTE: type checking yields errors as different layers have different num neurons
NetworkState = list[LayerEnsembleState]
NetworkCarry = tuple[list[LayerEnsembleCarry], PRNGKeyArray, PRNGKeyArray]
NetworkRecord = list[LayerEnsembleRecord]
NetworkRecordHistory = list[LayerEnsembleRecordHistory]
NetworkMonitor = LayerMonitor

DEFAULT_LAST_LAYER_OVERRIDE = {
    "neuron_config": {
        "high_pass_tau": None,
        "neuron_activation": "identity",
    },
    "input_wiring_args": {
        "recurrent_connect": 0.0,
    },
}

INPUT_EMBEDDINGS = ["scaled", "one_hot"]
OUTPUT_EMBEDDINGS = [
    "linear_layer",
]


@jaxtyped(typechecker=typechecker)
class ELMNetwork(eqx.Module, DynamicModel):
    num_input: int = eqx.field(static=True)
    num_output: int = eqx.field(static=True)
    num_hidden: int = eqx.field(static=True)
    num_layers: int = eqx.field(static=True)
    input_embedding: Optional[str] = eqx.field(static=True)
    input_embedding_args: dict = eqx.field(static=True)
    output_embedding: Optional[str] = eqx.field(static=True)
    output_embedding_args: dict = eqx.field(static=True)
    fused: bool = eqx.field(static=True)

    layers: list[ELMLayer]
    linear_embedding: Optional[eqx.nn.Linear]

    def __init__(
        self,
        num_input: int,
        num_output: int,
        num_hidden: int = 1024,
        num_layers: int = 2,
        input_embedding: Optional[str] = None,
        input_embedding_args: Optional[dict] = None,
        output_embedding: Optional[str] = None,
        output_embedding_args: Optional[dict] = None,
        layer_config: Optional[dict] = None,  # neuron_config becomes new default
        first_layer_override: Optional[dict] = None,  # ignored for num_layers = 1
        last_layer_override: Optional[dict] = None,
        fused: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `num_input`: The number of inputs or cardinality of input to the ELM network.
        - `num_output`: The number of outputs or cardinality of output of the ELM network.
            This will set the number of neurons in the last layer.
        - `num_hidden`: The number of neurons in the hidden layers of the ELM network.
            Parameter is ignored for single layered ELM networks.
        - `num_layers`: The number of total layers of the ELM networks.
        - `input_embedding`: The type of input embedding function.
        - `input_embedding_args`: Arguments to the input embedding function.
        - `output_embedding`: The type of output embedding function.
        - `output_embedding_args`: Arguments to the output embedding function.
        - `layer_config`: The new default configuration for all the layers.
        - `first_layer_override`: Additional configuration overrides applied only
            on the first layer of the ELM network (overriding `layer_config`).
            Note that this config is ignored for a single-layered network.
            Override chain: default <- layer config <- first-layer override
        - `last_layer_override`: Additional configuration overrides applied only
            on the last layer of the ELM network (overriding `layer_config`).
            Can manually override the number of neurons in the last layer.
            Note that the model first applies some default last-layer overrides
            before the herein specified values are used for override.
            Override chain: default <- layer config <- default l.l. override <- l.l. override
        - `fused`: Whether to run the sequence with fused Triton GPU kernels
            (see `fused_layer.FUSED_SETTINGS`).
        """
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.input_embedding = input_embedding
        self.input_embedding_args = (
            {} if input_embedding_args is None else input_embedding_args
        )
        self.output_embedding = output_embedding
        self.output_embedding_args = (
            {} if output_embedding_args is None else output_embedding_args
        )
        self.fused = fused
        _default_layer_config = (
            {} if layer_config is None else copy.deepcopy(layer_config)
        )
        _first_layer_override = (
            {} if first_layer_override is None else copy.deepcopy(first_layer_override)
        )
        _last_layer_override = (
            {} if last_layer_override is None else copy.deepcopy(last_layer_override)
        )

        layer_key, key = jrandom.split(key, 2)
        layers_key = jrandom.split(layer_key, self.num_layers)
        output_embed_key, key = jrandom.split(key, 2)
        input_embed_key, key = jrandom.split(key, 2)

        # basic sanity checks
        assert self.num_input > 0
        assert self.num_output > 0
        assert self.num_hidden > 0
        assert self.num_layers > 0
        assert self.num_layers > 1 or _first_layer_override == {}
        assert self.input_embedding is None or self.input_embedding in INPUT_EMBEDDINGS
        assert (
            self.output_embedding is None or self.output_embedding in OUTPUT_EMBEDDINGS
        )
        for override in (
            _first_layer_override,
            _last_layer_override,
            DEFAULT_LAST_LAYER_OVERRIDE,
        ):
            assert (
                "neuron_config" not in override
                or "delta_t" not in override["neuron_config"]
            )

        merged_layer_config = recursive_dict_merge(
            {"neuron_config": DEFAULT_ELM_LAYER_NEURON_CONFIG}, _default_layer_config
        )

        first_layer_override = copy.deepcopy(_first_layer_override)
        num_first_layer_neurons = first_layer_override.pop(
            "num_neuron", self.num_hidden
        )
        merged_first_layer_config = recursive_dict_merge(
            merged_layer_config, first_layer_override
        )

        last_layer_override = copy.deepcopy(_last_layer_override)
        num_last_layer_neurons = last_layer_override.pop("num_neuron", self.num_output)
        merged_last_layer_config = recursive_dict_merge(
            merged_layer_config, DEFAULT_LAST_LAYER_OVERRIDE
        )
        merged_last_layer_config = recursive_dict_merge(
            merged_last_layer_config, last_layer_override
        )

        # sequential layer creation
        layers = []
        layer_num_input = num_input
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                # treat like last layer for num_layers = 1
                config = merged_last_layer_config
                layer_num_neuron = num_last_layer_neurons
            elif i == 0:
                # only considered for num_layers > 1
                config = merged_first_layer_config
                layer_num_neuron = num_first_layer_neurons
            else:
                # only considered for num_layers > 2
                config = merged_layer_config
                layer_num_neuron = self.num_hidden

            layer = ELMLayer(
                **config,
                num_input=layer_num_input,
                num_neuron=layer_num_neuron,
                key=layers_key[i],
            )
            layers.append(layer)
            layer_num_input = layer_num_neuron
        self.layers = layers

        # output embedding layers
        if self.output_embedding == "linear_layer":
            self.linear_embedding = eqx.nn.Linear(
                in_features=layer_num_input,
                out_features=self.num_output,
                use_bias=True,
                key=output_embed_key,
                dtype=self.layers[0].ensemble.dtype,
            )
        else:
            self.linear_embedding = None

        # warnings for configuration
        if self.layers[-1].ensemble.neuron_activation != "identity":
            warnings.warn("Readout ELM layer has non-identity neuron activations.")
        if self.layers[-1].ensemble.high_pass_tau is not None:
            warnings.warn("Readout ELM layer has high-pass filtered neuron outputs.")
        if self.layers[-1].input_wiring_args["recurrent_connect"] > 0.0:
            warnings.warn("Readout ELM layer has recurrent connections.")
        if self.fused:
            from .fused_sequence import check_fused_network_support

            check_fused_network_support(self)

    def get_init_state(self) -> NetworkState:
        return [layer.get_init_state() for i, layer in enumerate(self.layers)]

    def get_init_carry(self, key: PRNGKeyArray) -> NetworkCarry:
        layers_key, dynamic_key, const_key = jrandom.split(key, num=3)
        layer_keys = jrandom.split(layers_key, num=self.num_layers)
        layer_ensemble_carries = []
        for i in range(self.num_layers):
            layer_ensemble_carries.append(
                self.layers[i].get_init_carry(key=layer_keys[i])
            )
        return layer_ensemble_carries, dynamic_key, const_key

    @eqx.filter_jit
    def embed_input(self, x: Integer[Array, ""]) -> Float[Array, "embed_dim"]:
        if self.input_embedding == "scaled":
            x = x * self.input_embedding_args["scale"]
        elif self.input_embedding == "one_hot":
            x = jnn.one_hot(x, self.num_input)
            if "scale" in self.input_embedding_args.keys():
                x = x * self.input_embedding_args["scale"]
        return x.astype(self.layers[0].ensemble.dtype)

    @eqx.filter_jit
    def dynamics(
        self,
        carry: NetworkCarry,
        x_t: Float[Array, "channel"],
        monitor: Optional[NetworkMonitor],
        inference: bool,
        branch_pools: Optional[list] = None,
    ) -> tuple[NetworkCarry, NetworkRecord]:
        layer_ensemble_carries, key, const_key = carry
        next_layer_ensemble_carries = []
        network_recording = []

        # loop over all layers for a single timestep
        for layer_idx in range(self.num_layers):

            # regular network step
            layer_ensemble_carry, layer_ensemble_recording = self.layers[
                layer_idx
            ].dynamics(
                layer_ensemble_carries[layer_idx],
                x_t,
                monitor,
                inference,
                branch_pools[layer_idx] if branch_pools is not None else None,
            )
            x_t = layer_ensemble_recording[0]["activity"]
            next_layer_ensemble_carries.append(layer_ensemble_carry)
            network_recording.append(layer_ensemble_recording)

        carry = (next_layer_ensemble_carries, key, const_key)
        return carry, network_recording

    @eqx.filter_jit
    def __call__(
        self,
        x: Input,
        monitor: Optional[NetworkMonitor] = None,
        inference: bool = False,
        init_carry: Optional[NetworkCarry] = None,
        grad_checkpointing: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> tuple[LayerActHistory, NetworkRecordHistory, NetworkCarry]:
        # input embedding
        x = eqx.filter_vmap(lambda x_t: self.embed_input(x_t))(x)

        # network inference
        carry = init_carry if init_carry is not None else self.get_init_carry(key=key)
        if self.fused:
            from .fused_sequence import fused_network_sequence

            carry, recording = fused_network_sequence(
                self, carry, x, monitor, inference
            )
        else:
            branch_pools = [layer.branch_pools for layer in self.layers]
            scan_fun = lambda carry, x_t: self.dynamics(
                carry, x_t, monitor, inference, branch_pools
            )
            if grad_checkpointing:
                scan_fun = jax.checkpoint(scan_fun)
            carry, recording = jlx.scan(scan_fun, carry, x)
        output = recording[-1][0]["activity"]

        # output embedding
        if self.output_embedding == "linear_layer":
            output = eqx.filter_vmap(lambda x_t: self.linear_embedding(x_t))(output)

        return output, recording, carry

    def get_train_params_filter(self) -> Any:
        elm_network_filter = jtu.tree_map(lambda x: eqx.is_inexact_array(x), self)
        elm_layer_filters = [layer.get_train_params_filter() for layer in self.layers]
        elm_network_filter = eqx.tree_at(
            lambda tree: (tree.layers,),
            elm_network_filter,
            replace=(elm_layer_filters,),
            is_leaf=lambda x: x is None,
        )
        return elm_network_filter
