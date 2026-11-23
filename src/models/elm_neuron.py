import math
from typing import Any, Callable, Optional, Union

import equinox as eqx
import jax.lax as jlx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray, jaxtyped
from typeguard import typechecked as typechecker

from ..datasets.neuronio.neuronio_data_utils import DEFAULT_Y_TRAIN_SOMA_SCALE
from .abstract_model import DynamicModel
from .elm_modeling_utils import (
    MACHINE_EPS,
    create_interlocking_indices,
    create_overlapping_window_indices,
    custom_tanh,
    inverse_scaled_sigmoid,
    scaled_sigmoid,
)
from .surrogate_gradients import fast_sigmoid

Input = Union[
    Float[Array, "time channel"], Bool[Array, "time channel"], Integer[Array, "time"]
]

NeuronActivity = Float[Array, "output"]
NeuronActHistory = Float[Array, "time output"]
NeuronState = dict[str, Float[Array, "..."]]
NeuronCarry = tuple[NeuronState, PRNGKeyArray, PRNGKeyArray]
NeuronRecord = dict[str, Float[Array, "..."]]
NeuronRecordHistory = dict[str, Float[Array, "time ..."]]
Monitor = list[str]

NEURON_MONITORS = ["branch", "mlp", "memory", "readout", "preact", "output"]
INPUT_ROUTINGS = ["random_routing", "neuronio_routing"]
MLP_ACTIVATIONS = ["relu", "relu_squared", "silu"]
INPUT_EMBEDDINGS = ["scaled", "one_hot"]
NEURON_ACTIVATIONS = ["identity", "relu", "spike"]


@jaxtyped(typechecker=typechecker)
class ELM(eqx.Module, DynamicModel):
    """The modified Expressive Leaky Memory (ELM) neuron model."""

    # key dimensions:
    num_input: int = eqx.field(static=True)
    num_output: int = eqx.field(static=True)
    num_memory: int = eqx.field(static=True)

    # synapses and branches:
    num_branch: int = eqx.field(static=True)
    num_synapse_per_branch: int = eqx.field(static=True)
    num_synapse: int = eqx.field(static=True)
    input_scale: float = eqx.field(static=True)
    input_to_synapse_routing: Optional[str] = eqx.field(static=True)
    branch_tau: Optional[float] = eqx.field(static=True)

    # integration and memory:
    mlp_num_input: int = eqx.field(static=True)
    mlp_num_hidden_layer: int = eqx.field(static=True)
    mlp_num_hidden_units: int = eqx.field(static=True)
    mlp_hidden_act: str = eqx.field(static=True)
    lambda_value: float = eqx.field(static=True)
    lambda_as_timescale_ratio: bool = eqx.field(static=True)
    memory_tau_min: float = eqx.field(static=True)
    memory_tau_max: float = eqx.field(static=True)
    memory_tau_learnable: bool = eqx.field(static=True)
    memory_tau_range_scale: float = eqx.field(static=True)

    # readout and activation:
    high_pass_tau: Optional[float] = eqx.field(static=True)
    neuron_activation: str = eqx.field(static=True)
    neuron_activation_fun: Callable = eqx.field(static=True)

    # neuronio required:
    neuronio_synapse_init: Optional[float] = eqx.field(static=True)
    neuronio_spike_bias: Optional[float] = eqx.field(static=True)

    # general setup:
    input_dropout: float = eqx.field(static=True)
    memory_dropout: float = eqx.field(static=True)
    input_embedding: Optional[str] = eqx.field(static=True)
    input_embedding_args: dict = eqx.field(static=True)
    delta_t: float = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    # learnable:
    _w_s: Float[Array, "{self.num_synapse}"]
    mlp: eqx.nn.MLP
    w_y: eqx.nn.Linear
    b: Float[Array, "{self.num_output}"]
    _tau_m: Float[Array, "{self.num_memory}"]

    # not learnable:
    _input_to_synapse_indices: Optional[Integer[Array, "{self.num_synapse}"]]
    _valid_indices_mask: Optional[Integer[Array, "{self.num_synapse}"]]

    def __init__(  # ReLU network defaults
        self,
        # key dimensions:
        num_input: int,
        num_output: int,
        num_memory: int = 100,
        # synapses and branches:
        num_branch: Optional[int] = None,
        num_synapse_per_branch: int = 1,
        input_scale: float = 1.0,
        input_to_synapse_routing: Optional[str] = None,
        branch_tau: Optional[float] = None,
        # integration and memory:
        mlp_num_hidden_layer: int = 1,
        mlp_num_hidden_units: Optional[int] = None,
        mlp_hidden_act: str = "relu_squared",
        lambda_value: float = 5.0,
        lambda_as_timescale_ratio: bool = True,
        memory_tau_min: float = 1.0,
        memory_tau_max: float = 1000.0,
        memory_tau_learnable: bool = False,
        memory_tau_range_scale: float = 10.0,
        # readout and filter:
        high_pass_tau: Optional[float] = None,
        neuron_activation: str = "identity",
        # neuronio required:
        neuronio_synapse_init: Optional[float] = None,
        neuronio_spike_bias: Optional[float] = None,
        # general setup:
        input_dropout: float = 0.0,
        memory_dropout: float = 0.0,
        input_embedding: Optional[str] = None,
        input_embedding_args: Optional[dict] = None,
        delta_t: float = 1.0,
        dtype: str = "float32",
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**:
        <!-- key dimensions: -->
        - `num_input`: The number of inputs or cardinality of input to the neuron.
        - `num_output`: The number of outputs or cardinality of output of the neuron.
        - `num_memory`: The number of memory units. The primary hidden state size
            of the recurrent cell. Used to keep track of information over time.
        <!-- synapses and branches: -->
        - `num_branch`: The number of neuron branches (also adjust
            `num_synapse_per_branch`). By default set to `num_input` and generally
            `num_branch * num_synapse_per_branch = num_input` must hold true.
        - `num_synapse_per_branch`: The number of synapses per neuron branch
            (also adjust `num_branch`). By default set to `1` and generally
            `num_branch * num_synapse_per_branch = num_input` must hold true.
            Primarily used for giving the neuron a large input dimensionality.
        - `input_scale`: The scaling factor applied to the input activations
        - `input_to_synapse_routing`: Options for compatibility with
            NeuronIO data format. With the `random_routing` option
            the input to synapse connections are sampled, and
            `num_branch * num_synapse_per_branch = num_input` can be violated.
        - `branch_tau`: The timescale (in ms) of the decay of the information in the
            branches. Works the same as individual current-based synapse dynamics.
            Primarily used for compensating temporally very sparse input.
        <!-- integration and memory: -->
        - `mlp_num_hidden_layer`: The number of hidden layers of the MLP. Setting it
            to zero will result in a linear layer integrating inputs with memory.
        - `mlp_num_hidden_units`: The number of hidden units per MLP hidden layer.
            Per default twice the `num_memory` value. Greater values let the
            neuron learn more sophisticated input-to-output transformations.
        - `mlp_hidden_act`: The hidden layer activation function of the MLP.
        - `lambda_value`: The value of the lambda model parameter. The factor
            configuring how much faster input timescales are than forget timescales.
            Higher values allow for faster absorption of new information.
        - `lambda_as_timescale_ratio`: Whether to use lambda directly as the
            input to memory timescale ratio instead of scaling the memory update.
        - `memory_tau_min`: Smallest timescale (in ms) to initialize the memory
            timescales with, and is also used to determine the learnable lower-bound
        - `memory_tau_max`: Largest timescale (in ms) to initialize the memory
            timescales with (equidistant in log-space), and is also used to
            determine the learnable upper-bound. Greater values / slower timescales
            give the model longer-range temporal processing abilities.
        - `memory_tau_range_scale`: Scales learnable range based on the init range.
        - `memory_tau_learnable`: Whether to learn the memory timescales.
        <!-- readout and filter: -->
        - `high_pass_tau`: The timescale used for the neuron high-pass filter.
            Works by computing an exponential moving average of neuron readout.
            The EMA is then subtracted from neuron readout for neuron output.
        - `neuron_activation`: The neuron's activations e.g. `relu` or `spike`.
        <!-- neuronio required: -->
        - `neuronio_synapse_init`: Initialize synapses as constant before scaling.
            During training the synapses are rectified using relu to stay positive.
        - `neuronio_spike_bias`: Initialize bias value for spike output on neuronio.
         <!-- general setup: -->
        - `input_dropout`: The variational dropout (probability) for the input.
        - `memory_dropout`: The variational dropout probability for the memory update.
        - `input_embedding`: The type of input embedding function to use
        - `input_embedding_args`: Arguments to the input embedding function.
        - `delta_t`: The time (in ms) between two successive inputs to the neuron.
        - `dtype`: Which dtype learnable weights will be initialized with.
        """
        super(ELM, self).__init__()
        mlp_key, w_s_key, w_y_key, routing_key, key = jrandom.split(key, 5)

        # key dimensions:
        self.num_input, self.num_output = num_input, num_output
        self.num_memory = num_memory
        # synapses and branches:
        self.num_synapse_per_branch = num_synapse_per_branch
        self.input_scale = input_scale
        self.input_to_synapse_routing = input_to_synapse_routing
        self.branch_tau = branch_tau
        # integration and memory:
        self.mlp_num_hidden_layer = mlp_num_hidden_layer
        self.mlp_hidden_act = mlp_hidden_act
        self.lambda_value = lambda_value
        self.lambda_as_timescale_ratio = lambda_as_timescale_ratio
        self.memory_tau_min = memory_tau_min
        self.memory_tau_max = memory_tau_max
        self.memory_tau_learnable = memory_tau_learnable
        self.memory_tau_range_scale = memory_tau_range_scale
        # readout and filter:
        self.high_pass_tau = high_pass_tau
        self.neuron_activation = neuron_activation
        # neuronio required:
        self.neuronio_synapse_init = neuronio_synapse_init
        self.neuronio_spike_bias = neuronio_spike_bias
        # general setup:
        self.input_dropout = input_dropout
        self.memory_dropout = memory_dropout
        self.input_embedding = input_embedding
        self.input_embedding_args = (
            {} if input_embedding_args is None else input_embedding_args
        )
        self.delta_t = delta_t
        self.dtype = dtype

        # derived properties
        self.mlp_num_hidden_units = (
            mlp_num_hidden_units if mlp_num_hidden_units is not None else 2 * num_memory
        )
        self.num_branch = num_branch if num_branch is not None else num_input
        self.mlp_num_input = self.num_branch + num_memory
        self.num_synapse = num_synapse_per_branch * self.num_branch

        # basic assertions
        self.perform_sanity_checks()

        # initialization of synapse weights
        syn_min, syn_max = -1.0, 1.0
        if neuronio_synapse_init is not None:
            syn_min = syn_max = neuronio_synapse_init
        syn_scale = math.sqrt(num_synapse_per_branch)
        self._w_s = jrandom.uniform(
            key=w_s_key,
            shape=(self.num_synapse,),
            minval=syn_min / syn_scale,
            maxval=syn_max / syn_scale,
            dtype=self.dtype,
        )

        # initialization of mlp weights
        if self.mlp_hidden_act == "relu":
            mlp_hidden_act_fun = lambda x: jnn.relu(x)
        elif self.mlp_hidden_act == "silu":
            mlp_hidden_act_fun = lambda x: jnn.silu(x)
        elif self.mlp_hidden_act == "relu_squared":
            mlp_hidden_act_fun = lambda x: jnp.square(jnn.relu(x))
        else:
            raise NotImplementedError
        self.mlp = eqx.nn.MLP(
            in_size=self.mlp_num_input,
            out_size=num_memory,
            width_size=self.mlp_num_hidden_units,
            depth=mlp_num_hidden_layer,
            activation=mlp_hidden_act_fun,
            key=mlp_key,
            dtype=self.dtype,
        )

        # initialization of memory timescales
        _tau_m = jnp.logspace(
            math.log10(self.memory_tau_min),
            math.log10(self.memory_tau_max),
            num_memory,
            dtype=self.dtype,
        )
        self._tau_m = inverse_scaled_sigmoid(
            _tau_m,
            self.memory_tau_min / self.memory_tau_range_scale,
            self.memory_tau_max * math.sqrt(self.memory_tau_range_scale),
        )

        # initialization of readout weights
        w_y = eqx.nn.Linear(
            in_features=self.num_memory,
            out_features=num_output,
            use_bias=False,
            key=w_y_key,
            dtype=self.dtype,
        )
        self.w_y = w_y

        # initialization of output bias
        b = jnp.zeros((self.num_output,), dtype=self.dtype)
        if self.neuronio_spike_bias is not None:
            assert self.num_output == 2
            b = b.at[0].set(self.neuronio_spike_bias)
        self.b = b

        # define the neuron activation function
        if self.neuron_activation == "identity":
            self.neuron_activation_fun = lambda x: x
        elif self.neuron_activation == "relu":
            self.neuron_activation_fun = lambda x: jnn.relu(x)
        elif self.neuron_activation == "spike":
            self.neuron_activation_fun = lambda x: fast_sigmoid(x)
        else:
            raise NotImplementedError

        # neuronio related input routing
        self._input_to_synapse_indices, self._valid_indices_mask = (
            self.create_input_to_synapse_indices(routing_key)
        )

    @property
    @jaxtyped(typechecker=typechecker)
    def w_s(self) -> Float[Array, "{self.num_synapse}"]:
        if self.neuronio_synapse_init is not None:
            return jnn.relu(self._w_s)
        return self._w_s

    @property
    @jaxtyped(typechecker=typechecker)
    def tau_m(self) -> Float[Array, "{self.num_memory}"]:
        tau_m = scaled_sigmoid(
            self._tau_m,
            self.memory_tau_min / self.memory_tau_range_scale,
            self.memory_tau_max * math.sqrt(self.memory_tau_range_scale),
        )
        return tau_m if self.memory_tau_learnable else jlx.stop_gradient(tau_m)

    @property
    @jaxtyped(typechecker=typechecker)
    def kappa_b(self) -> Optional[Float[Array, ""]]:
        if self.branch_tau is not None:
            return jnp.exp(-self.delta_t / jnp.clip(self.branch_tau, min=MACHINE_EPS))
        else:
            return None

    @property
    @jaxtyped(typechecker=typechecker)
    def kappa_m(self) -> Float[Array, "{self.num_memory}"]:
        return jnp.exp(-self.delta_t / jnp.clip(self.tau_m, min=MACHINE_EPS))

    @property
    @jaxtyped(typechecker=typechecker)
    def kappa_r(self) -> Optional[Float[Array, ""]]:
        if self.high_pass_tau is not None:
            return jnp.exp(
                -self.delta_t / jnp.clip(self.high_pass_tau, min=MACHINE_EPS)
            )
        else:
            return None

    @property
    @jaxtyped(typechecker=typechecker)
    def kappa_lambda(self) -> Float[Array, "{self.num_memory}"]:
        if self.lambda_as_timescale_ratio:
            return jnp.exp(
                -self.delta_t
                * self.lambda_value
                / jnp.clip(self.tau_m, min=MACHINE_EPS)
            )
        else:
            return jnp.exp(-self.delta_t / jnp.clip(self.tau_m, min=MACHINE_EPS))

    @property
    @jaxtyped(typechecker=typechecker)
    def input_to_synapse_indices(
        self,
    ) -> Optional[Integer[Array, "{self.num_synapse}"]]:
        return jlx.stop_gradient(self._input_to_synapse_indices)

    @property
    @jaxtyped(typechecker=typechecker)
    def valid_indices_mask(self) -> Optional[Integer[Array, "{self.num_synapse}"]]:
        return jlx.stop_gradient(self._valid_indices_mask)

    @jaxtyped(typechecker=typechecker)
    def create_input_to_synapse_indices(self, key: PRNGKeyArray) -> tuple[
        Optional[Integer[Array, "{self.num_synapse}"]],
        Optional[Integer[Array, "{self.num_synapse}"]],
    ]:
        if self.input_to_synapse_routing == "random_routing":
            # randomly select num_synapse from num_input
            input_to_synapse_indices = jrandom.randint(
                key=key,
                shape=(self.num_synapse,),
                minval=0,
                maxval=self.num_input,
            )
            valid_indices_mask = jnp.ones_like(input_to_synapse_indices)
            return input_to_synapse_indices, valid_indices_mask
        elif self.input_to_synapse_routing == "neuronio_routing":
            # sanity check of input configuration
            assert (
                math.ceil(self.num_input / self.num_branch)
                <= self.num_synapse_per_branch
            )

            # interlace excitatory and inhibitory inputs
            interlocking_indices = create_interlocking_indices(self.num_input)
            # assign neighbouring inputs to same branch
            overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
                self.num_input, self.num_branch, self.num_synapse_per_branch
            )
            input_to_synapse_indices = interlocking_indices[overlapping_indices]

            return input_to_synapse_indices, valid_indices_mask
        else:
            return None, None

    @jaxtyped(typechecker=typechecker)
    def route_input_to_synapses(
        self, x: Float[Array, "time dim"]
    ) -> Float[Array, "time {self.num_synapse}"]:
        if self.input_to_synapse_routing is not None:
            x = x[:, self.input_to_synapse_indices] * self.valid_indices_mask
        return x

    @jaxtyped(typechecker=typechecker)
    def get_init_state(self) -> NeuronState:
        neuron_state = {"m_t": jnp.zeros((self.num_memory,), dtype=self.dtype)}
        if self.branch_tau is not None:
            neuron_state["b_t"] = jnp.zeros((self.num_branch,), dtype=self.dtype)
        if self.high_pass_tau is not None:
            neuron_state["r_t"] = jnp.zeros((self.num_output,), dtype=self.dtype)
        return neuron_state

    @jaxtyped(typechecker=typechecker)
    def get_init_carry(self, key: PRNGKeyArray) -> NeuronCarry:
        dynamic_key, const_key = jrandom.split(key, 2)
        state = self.get_init_state()
        return state, dynamic_key, const_key

    @jaxtyped(typechecker=typechecker)
    def apply_input_dropout(
        self,
        activity: Float[Array, "{self.num_synapse}"],
        inference: bool,
        dropout_key: PRNGKeyArray,
    ) -> Float[Array, "{self.num_synapse}"]:
        if self.input_dropout > 0.0 and not inference:
            dropout_mask = jrandom.bernoulli(
                key=dropout_key,
                p=self.input_dropout,
                shape=activity.shape,
            )
            # scaling preserves mean
            activity = jnp.where(dropout_mask, 0, activity) / (1 - self.input_dropout)
        return activity

    @jaxtyped(typechecker=typechecker)
    def apply_memory_dropout(
        self,
        mlp_out: Float[Array, "{self.num_memory}"],
        inference: bool,
        const_key: PRNGKeyArray,
    ) -> Float[Array, "{self.num_memory}"]:
        if self.memory_dropout > 0.0 and not inference:
            memory_dropout_mask = jrandom.bernoulli(
                key=const_key,
                p=self.memory_dropout,
                shape=mlp_out.shape,
            )
            # scaling preserves std for zero mean
            mlp_out = jnp.where(memory_dropout_mask, 0.0, mlp_out) * jlx.rsqrt(
                1 - self.memory_dropout
            )
        return mlp_out

    @jaxtyped(typechecker=typechecker)
    def branch_integration(
        self,
        x_t: Float[Array, "{self.num_synapse}"],
        w_s: Float[Array, "{self.num_synapse}"],
    ) -> Float[Array, "{self.num_branch}"]:
        s_inp = w_s * x_t
        s_inp = jnp.reshape(s_inp, (self.num_branch, self.num_synapse_per_branch))
        b_inp = jnp.sum(s_inp, axis=-1) * self.input_scale
        return b_inp

    @jaxtyped(typechecker=typechecker)
    def apply_branch_decay(
        self,
        b_act: Float[Array, "{self.num_branch}"],
        prev_state: NeuronState,
        next_state: NeuronState,
        kappa_b: Optional[Float[Array, ""]],
    ) -> tuple[Float[Array, "{self.num_branch}"], NeuronState]:
        if self.branch_tau is not None:
            next_state["b_t"] = kappa_b * prev_state["b_t"] + b_act
            b_act = next_state["b_t"]

        return b_act, next_state

    @jaxtyped(typechecker=typechecker)
    def memory_update(
        self,
        m_t_decayed: Float[Array, "{self.num_memory}"],
        mlp_out: Float[Array, "{self.num_memory}"],
        kappa_lambda: Float[Array, "{self.num_memory}"],
    ) -> Float[Array, "{self.num_memory}"]:
        delta_m_t = custom_tanh(mlp_out)
        if self.lambda_as_timescale_ratio:
            return m_t_decayed + (1 - kappa_lambda) * delta_m_t
        else:
            return m_t_decayed + self.lambda_value * (1 - kappa_lambda) * delta_m_t

    @jaxtyped(typechecker=typechecker)
    def apply_high_pass(
        self,
        y_t: Float[Array, "{self.num_output}"],
        prev_state: NeuronState,
        next_state: NeuronState,
        kappa_r: Optional[Float[Array, ""]],
    ) -> tuple[Float[Array, "{self.num_output}"], NeuronState]:
        if self.high_pass_tau is not None:
            next_state["r_t"] = kappa_r * prev_state["r_t"] + (1 - kappa_r) * y_t
            y_t = y_t - next_state["r_t"]

        return y_t, next_state

    @eqx.filter_jit
    def embed_input(self, x: Integer[Array, ""]) -> Float[Array, "embed_dim"]:
        if self.input_embedding == "scaled":
            x = x * self.input_embedding_args["scale"]
        elif self.input_embedding == "one_hot":
            x = jnn.one_hot(x, self.num_input)
            if "scale" in self.input_embedding_args.keys():
                x = x * self.input_embedding_args["scale"]
        x = x.astype(self.dtype)
        return x

    # all "apply" functions are potentially no-op
    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def dynamics(
        self,
        carry: NeuronCarry,
        x_t: Float[Array, "_"],
        monitor: Optional[Monitor],
        inference: bool,
        precomputed_branches: bool = False,
    ) -> tuple[NeuronCarry, NeuronRecord]:
        monitor = monitor if monitor else []
        prev_state, dynamic_key, const_key = carry
        next_state: NeuronState = {}

        # const key for variational dropout
        input_dropout_key, memory_dropout_key = jrandom.split(const_key, 2)

        # integrate synaptic input on branches (or take precomputed branch sums)
        if precomputed_branches:
            assert self.input_dropout == 0.0
            b_act = x_t
        else:
            x_t = self.apply_input_dropout(x_t, inference, input_dropout_key)
            b_act = self.branch_integration(x_t, self.w_s)
        b_act, next_state = self.apply_branch_decay(
            b_act, prev_state, next_state, self.kappa_b
        )

        # integrate branch input with memory
        m_t_decayed = self.kappa_m * prev_state["m_t"]
        mlp_out = self.mlp(jnp.concatenate([b_act, m_t_decayed], axis=-1))
        mlp_out = self.apply_memory_dropout(mlp_out, inference, memory_dropout_key)
        next_state["m_t"] = self.memory_update(m_t_decayed, mlp_out, self.kappa_lambda)

        # generate output
        readout_t = self.w_y(next_state["m_t"])
        pre_act, next_state = self.apply_high_pass(
            readout_t, prev_state, next_state, self.kappa_r
        )
        y_t = self.neuron_activation_fun(pre_act + self.b)

        # construct carry and records
        carry = (next_state, dynamic_key, const_key)
        recording = {"output": y_t}
        if "branch" in monitor:
            recording["branch"] = b_act
        if "mlp" in monitor:
            recording["mlp"] = mlp_out
        if "memory" in monitor:
            recording["memory"] = next_state["m_t"]
        if "readout" in monitor:
            recording["readout"] = readout_t
        if "preact" in monitor:
            recording["preact"] = pre_act
        return carry, recording

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self,
        x: Input,
        monitor: Optional[Monitor] = None,
        inference: bool = False,
        init_carry: Optional[NeuronCarry] = None,
        *,
        key: PRNGKeyArray,
    ) -> tuple[NeuronActHistory, NeuronRecordHistory, NeuronCarry]:
        # input embedding
        x = eqx.filter_vmap(lambda x_t: self.embed_input(x_t))(x)

        # neuron inference
        x = self.route_input_to_synapses(x)
        scan_fun = lambda carry_state, input_spikes: self.dynamics(
            carry_state, input_spikes, monitor, inference
        )
        carry = init_carry if init_carry is not None else self.get_init_carry(key=key)
        carry, recording = jlx.scan(scan_fun, carry, x)

        return recording["output"], recording, carry

    @eqx.filter_jit
    @jaxtyped(typechecker=typechecker)
    def neuronio_eval_forward(
        self,
        x: Input,
        monitor: Optional[Monitor] = None,
        y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
        init_carry: Optional[NeuronCarry] = None,
        *,
        key: PRNGKeyArray,
    ) -> tuple[NeuronActHistory, NeuronRecordHistory, NeuronCarry]:
        x = x.astype(self.dtype)
        output, recording, carry = self.__call__(
            x, monitor=monitor, inference=True, init_carry=init_carry, key=key
        )
        spike_pred, soma_pred = output[..., 0], output[..., 1]

        # apply sigmoid to spike (probability) prediction
        spike_pred = jnn.sigmoid(spike_pred)
        # apply soma scale to soma prediction
        soma_pred = 1 / y_train_soma_scale * soma_pred

        output = jnp.stack([spike_pred, soma_pred], axis=-1)
        return output, recording, carry

    def get_train_params_filter(self) -> Any:
        elm_filter = jtu.tree_map(lambda x: eqx.is_inexact_array(x), self)
        elm_filter = eqx.tree_at(
            lambda tree: (
                tree._tau_m,
                tree._input_to_synapse_indices,
                tree._valid_indices_mask,
            ),
            elm_filter,
            replace=(
                self.memory_tau_learnable,
                False,
                False,
            ),
            is_leaf=lambda x: x is None,
        )
        return elm_filter

    def perform_sanity_checks(self):
        assert self.num_input > 0
        assert self.num_output > 0
        assert self.num_memory > 0
        assert self.num_branch > 0
        assert self.num_synapse_per_branch > 0
        assert self.input_scale > 0.0
        assert (self.input_to_synapse_routing is None) or (
            self.input_to_synapse_routing in INPUT_ROUTINGS
        )
        assert (self.num_synapse == self.num_input) or (
            self.input_to_synapse_routing is not None
        )
        assert self.branch_tau is None or self.branch_tau > 0
        assert self.mlp_num_hidden_layer >= 0
        assert self.mlp_num_hidden_units > 0
        assert self.mlp_hidden_act in MLP_ACTIVATIONS
        assert self.lambda_value > 0
        assert 0 < self.memory_tau_min <= self.memory_tau_max
        assert self.memory_tau_range_scale >= 1.0
        assert self.high_pass_tau is None or self.high_pass_tau > 0
        assert self.neuron_activation in NEURON_ACTIVATIONS
        assert 0.0 <= self.memory_dropout < 1.0
        assert 0.0 <= self.input_dropout < 1.0
        assert self.input_embedding is None or self.input_embedding in INPUT_EMBEDDINGS
        assert self.delta_t > 0


def calc_approx_num_params(
    num_output: int,
    num_memory: int,
    num_branch: int,
    num_synapse_per_branch: int = 1,
    mlp_num_hidden_layer: int = 1,
    mlp_num_hidden_units: int = None,
) -> int:
    """Only counting trainable parameters."""

    if mlp_num_hidden_units is None:
        mlp_num_hidden_units = 2 * num_memory
    synapse_params = num_branch * num_synapse_per_branch
    if mlp_num_hidden_layer == 0:
        mlp_params = num_memory * (num_memory + num_branch + 1)
    elif mlp_num_hidden_layer == 1:
        mlp_params = mlp_num_hidden_units * (num_memory + num_branch + 1)
        mlp_params += num_memory * (mlp_num_hidden_units + 1)
    else:
        raise NotImplementedError
    readout_params = num_output * (num_memory + 1)
    return synapse_params + mlp_params + readout_params
