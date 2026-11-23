"""Opt-in fused GPU kernels for ELM layers (Triton, called from JAX through jax-triton).

This module holds the layer side: the settings and their checks (FUSED_SETTINGS,
check_fused_support), the layer description (FusedSpec), the piece sizes, and the
parameter preparation. The kernels are in triton_kernels.py, their launching in
triton_launch.py, and the time loop around them in fused_sequence.py: the whole sequence
runs fused (ELMNetwork(fused=True)).

Settings a layer uses that the kernels don't implement are rejected when the network is
built (check_fused_support).
"""

import dataclasses
import functools
from typing import Optional

import equinox as eqx
import jax
import jax.lax as jlx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from .elm_modeling_utils import MACHINE_EPS
from .surrogate_gradients import DEFAULT_SCALE, DEFAULT_THRESHOLD

F32 = jnp.float32
MIN_LANES = 8  # batch lanes (padded): tl.dot needs every dim >= 8
INGRAD_TILE = 1024  # synapse entries x lanes per gather in the input-grad kernel
MAX_LANES = 64  # batch lanes per program, at most; larger batches are split into groups
MAX_MLP_CHUNK = 64  # rows of the mlp's first weight matrix per piece, at most
MAX_SYNAPSE_CHUNK = 8  # synapses per branch gathered at once (random wiring), at most
SUPPORTED_ACTIVATIONS = ("identity", "relu", "spike")
SUPPORTED_MLP_ACTIVATIONS = ("relu", "relu_squared", "silu")


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


@dataclasses.dataclass(frozen=True)
class FusedSpec:
    """Static description of one fused layer step (hashable)."""

    num_input: int  # layer input channels (I)
    num_neuron: int  # N
    num_branch: int  # NB
    num_syn: int  # synapses per branch (P)
    num_memory: int  # M
    mlp_depth: int  # 0 (linear map) or 1 (one hidden layer)
    mlp_hidden: int  # H
    mlp_act: str  # 'relu', 'relu_squared', or 'silu'
    gain: float  # input_scale
    high_pass: bool  # high_pass_tau is not None
    kappa_r: float  # high-pass EMA decay (0.0 when high_pass is False)
    activation: str  # 'identity', 'relu', or 'spike'
    rec_output: bool = False
    rec_preact: bool = False
    rec_readout: bool = False
    rec_mlp: bool = False
    rec_branch: bool = False
    interpret: bool = False
    num_warps: int = 4
    mlp_chunk: int = MAX_MLP_CHUNK  # fused_kernels_args hidden_chunk
    max_synapse_chunk: int = MAX_SYNAPSE_CHUNK
    max_lanes: int = MAX_LANES
    spike_threshold: float = DEFAULT_THRESHOLD
    spike_scale: float = DEFAULT_SCALE
    # tl.dot precision; None (always, from the settings): jax_default_matmul_precision,
    # like XLA's own dots (tests may set it)
    precision: Optional[str] = None
    # pooled wiring (random_pooled, branch pools): sharers per list; 0: random wiring
    pool_size: int = 0
    # fast_backward (sequence path, pooled layers): the weight gradients are summed over
    # windows of this many steps, from inputs kept in the backward, and the forward's
    # branch sums are kept; 1 (default): per step
    grad_window: int = 1

    @property
    def fast_backward(self) -> bool:
        return self.grad_window > 1

    @property
    def num_runs(self) -> int:  # groups of neurons per branch
        return -(-self.num_neuron // self.pool_size)

    @property
    def gp(self) -> int:  # padded group size (a group has at most min(G, N) neurons)
        return next_pow2(min(self.pool_size, self.num_neuron))

    @property
    def num_lists(self) -> int:  # one shared list per branch and group
        return self.num_branch * self.num_runs

    @property
    def nbp(self) -> int:
        return next_pow2(self.num_branch)

    @property
    def pp(self) -> int:
        return next_pow2(self.num_syn)

    @property
    def mp(self) -> int:
        return next_pow2(self.num_memory)

    @property
    def hp(self) -> int:
        return next_pow2(self.mlp_hidden) if self.mlp_depth == 1 else 1

    @property
    def num_slots(self) -> int:
        return self.num_input + self.num_neuron

    @property
    def num_synapse(self) -> int:
        return self.num_branch * self.num_syn

    def weight_keys(self) -> tuple:
        mlp = ("w1b", "w1m", "b1", "w2") if self.mlp_depth == 1 else ("w2b", "w2m")
        return ("conn", "ws") + mlp + ("b2", "wy", "km", "kl", "bias")

    def grad_keys(self) -> tuple:
        return tuple(k for k in self.weight_keys() if k != "conn")


########## LAYER INTEGRATION HELPERS ##########


def _fused_args(layer) -> dict:
    return dict(layer.fused_kernels_args)


# The kernel settings (a layer's fused_kernels_args), all optional, with their defaults.
FUSED_SETTINGS = {
    "max_lanes": MAX_LANES,  # sequences per program, at most (a power of two, >= 8)
    "num_warps": 4,  # warps per program (Triton's num_warps)
    "hidden_chunk": MAX_MLP_CHUNK,  # mlp rows per piece, at most (a power of two, >= 8)
    "max_synapse_chunk": MAX_SYNAPSE_CHUNK,  # random wiring: synapses per branch per piece
    "fast_backward": 1,  # gradient window: weight gradients summed over this many steps
    "interpret": None,  # tests: Triton's CPU interpreter; None: compiled on a gpu
}


def fused_settings(layer) -> dict:
    """The layer's kernel settings: FUSED_SETTINGS updated with its fused_kernels_args."""
    args = _fused_args(layer)
    return {key: args.get(key, default) for key, default in FUSED_SETTINGS.items()}


def check_fused_support(layer) -> None:
    """Asserts that a layer only uses settings the fused kernels implement."""
    ens = layer.ensemble
    msg = "fused: unsupported setting "
    args = _fused_args(layer)
    unknown = set(args) - set(FUSED_SETTINGS)
    assert not unknown, (
        msg + f"fused_kernels_args {sorted(unknown)} (known: {sorted(FUSED_SETTINGS)})"
    )
    st = fused_settings(layer)
    is_int = lambda v: isinstance(v, int) and not isinstance(v, bool)
    pow2 = lambda v, low: is_int(v) and v >= low and v & (v - 1) == 0
    assert pow2(st["max_lanes"], MIN_LANES), msg + "max_lanes (a power of two, >= 8)"
    assert pow2(st["num_warps"], 1), msg + "num_warps (a power of two)"
    assert pow2(st["hidden_chunk"], MIN_LANES), (
        msg + "hidden_chunk (a power of two, >= 8)"
    )
    assert pow2(st["max_synapse_chunk"], 1), msg + "max_synapse_chunk (a power of two)"
    assert is_int(st["fast_backward"]) and st["fast_backward"] >= 1, (
        msg + "fast_backward (a window of steps: 1, or a divisor of the sequence length)"
    )
    assert st["interpret"] is None or isinstance(st["interpret"], bool), (
        msg + "interpret (True, False, or None: compiled on a gpu)"
    )
    share = layer.input_wiring_args.get("share_size", 16)
    pooled = layer.input_pool_order is not None and share > 1
    assert st["fast_backward"] == 1 or pooled, (
        msg + "fast_backward (needs pooled wiring, random_pooled with share_size > 1)"
    )
    # neuron settings
    assert ens.dtype == "float32", msg + "dtype (float32 only)"
    assert ens.num_output == 1, msg + "num_output"
    assert ens.neuron_activation in SUPPORTED_ACTIVATIONS, msg + "neuron_activation"
    assert ens.mlp_num_hidden_layer in (0, 1), msg + "mlp_num_hidden_layer (0 or 1)"
    assert (
        ens.mlp_num_hidden_layer == 0
        or ens.mlp_hidden_act in SUPPORTED_MLP_ACTIVATIONS
    ), msg + "mlp_hidden_act"
    assert ens.branch_tau is None, msg + "branch_tau"
    assert ens.memory_dropout == 0.0, msg + "memory_dropout"
    assert ens.input_dropout == 0.0, msg + "neuron input_dropout"
    assert ens.input_to_synapse_routing is None, msg + "input_to_synapse_routing"
    assert ens.lambda_as_timescale_ratio, msg + "lambda_as_timescale_ratio"
    assert ens.neuronio_spike_bias is None, msg + "neuronio_spike_bias"
    assert ens.input_embedding is None, msg + "neuron input_embedding"


def layer_fused_spec(layer, ensemble_monitor=(), **overrides) -> FusedSpec:
    ens = layer.ensemble
    st = fused_settings(layer)
    interpret = st["interpret"]
    if interpret is None:  # the kernels compile for gpus; elsewhere, the interpreter
        interpret = jax.default_backend() != "gpu"
    share = layer.input_wiring_args.get("share_size", 16)
    pooled = layer.input_pool_order is not None and share > 1
    window = st["fast_backward"]
    assert window == 1 or pooled, "fast_backward: needs pooled wiring (random_pooled)"
    high_pass = ens.high_pass_tau is not None
    kappa_r = (
        float(
            np.exp(
                np.float32(-ens.delta_t)
                / np.maximum(np.float32(ens.high_pass_tau), np.float32(MACHINE_EPS))
            )
        )
        if high_pass
        else 0.0
    )
    kwargs = dict(
        num_input=layer.num_input,
        num_neuron=layer.num_neuron,
        num_branch=ens.num_branch,
        num_syn=ens.num_synapse_per_branch,
        num_memory=ens.num_memory,
        mlp_depth=ens.mlp_num_hidden_layer,
        mlp_hidden=ens.mlp_num_hidden_units,
        mlp_act=ens.mlp_hidden_act,
        gain=float(ens.input_scale),
        high_pass=high_pass,
        kappa_r=kappa_r,
        activation=ens.neuron_activation,
        rec_output="output" in ensemble_monitor,
        rec_preact="preact" in ensemble_monitor,
        rec_readout="readout" in ensemble_monitor,
        rec_mlp="mlp" in ensemble_monitor,
        rec_branch="branch" in ensemble_monitor,
        interpret=bool(interpret),
        num_warps=st["num_warps"],
        mlp_chunk=st["hidden_chunk"],
        max_synapse_chunk=st["max_synapse_chunk"],
        max_lanes=st["max_lanes"],
        pool_size=share if pooled else 0,
        grad_window=window,
    )
    kwargs.update(overrides)
    spec = FusedSpec(**kwargs)
    assert not spec.fast_backward or len(mlp_chunks(spec)) <= _MAX_UNROLLED_CHUNKS, (
        f"fast_backward: at most {_MAX_UNROLLED_CHUNKS} mlp pieces (a larger hidden_chunk)"
    )
    return spec


def layer_step_params(layer) -> dict:
    """Unpadded step parameters read from the layer's ensemble."""
    ens = layer.ensemble
    w_s, km, kl = eqx.filter_vmap(
        lambda neuron: (neuron.w_s, neuron.kappa_m, neuron.kappa_lambda)
    )(ens)

    def bias_or_zeros(linear):
        if linear.bias is not None:
            return linear.bias
        return jnp.zeros(linear.weight.shape[:2], dtype=linear.weight.dtype)

    params = {
        "conn": layer.input_connectome,
        "w_s": w_s,
        "km": km,
        "kl": kl,
        "bias": ens.b[:, 0],  # (neuron,)
        "wy": ens.w_y.weight,  # (neuron, 1, num_memory)
    }
    if layer.input_pool_order is not None:
        params["pool_order"] = layer.input_pool_order
    mlp_layers = ens.mlp.layers
    if ens.mlp_num_hidden_layer == 1:
        params["w1"] = mlp_layers[0].weight
        params["b1"] = bias_or_zeros(mlp_layers[0])
        params["w2"] = mlp_layers[1].weight
        params["b2"] = bias_or_zeros(mlp_layers[1])
    else:
        params["w0"] = mlp_layers[0].weight
        params["b0"] = bias_or_zeros(mlp_layers[0])
    return params


########## PARAMETER PREPARATION (ONCE PER SEQUENCE) ##########


def prepare_fused_params(spec: FusedSpec, params: dict) -> dict:
    """Splits / reshapes parameters (unpadded) and builds the inverse wiring."""
    n, nb, p, m = spec.num_neuron, spec.num_branch, spec.num_syn, spec.num_memory
    nbp, s = spec.nbp, spec.num_synapse
    conn = params["conn"].astype(jnp.int32)
    fp = {
        "conn": conn.reshape(n, nb, p),
        "ws": params["w_s"].reshape(n, nb, p),
        "km": params["km"],
        "kl": params["kl"],
        "wy": params["wy"].reshape(n, m),
        "bias": params["bias"],
    }
    if spec.mlp_depth == 1:
        w1 = params["w1"]
        fp["w1b"] = w1[:, :, :nb]
        fp["w1m"] = w1[:, :, nb:]
        fp["b1"] = params["b1"]
        fp["w2"] = params["w2"]
        fp["b2"] = params["b2"]
    else:
        w0 = params["w0"]
        fp["w2b"] = w0[:, :, :nb]
        fp["w2m"] = w0[:, :, nb:]
        fp["b2"] = params["b0"]

    if spec.pool_size:  # pooled wiring: the shared lists' tables (no per-synapse ones)
        fp.update(_pool_tables(spec, conn, params["pool_order"]))
        return fp
    # inverse wiring (csr): synapses sorted by the input slot they read from
    flat = conn.reshape(-1)
    order = jnp.argsort(flat, stable=True).astype(jnp.int32)
    counts = jnp.bincount(flat, length=spec.num_slots).astype(jnp.int32)
    fp["inv_start"] = (jnp.cumsum(counts) - counts).astype(jnp.int32)
    fp["inv_count"] = counts
    neuron_idx, syn_idx = order // s, order % s
    # index into the (padded) per-branch cotangent buffer of the backward kernel
    fp["inv_nb"] = (neuron_idx * nbp + syn_idx // p).astype(jnp.int32)
    # only used by the backward kernel (zero cotangent by construction)
    fp["inv_w"] = jlx.stop_gradient(params["w_s"].reshape(-1)[order])
    return fp


# tests only: False keeps every weight copy and history in float32, to check that the
# bf16 storage (bf16_weight_keys, bf16_history_keys) is bit-identical on a gpu
_BF16_STORAGE = True


def _bf16_dots(precision) -> bool:
    """Whether dots at this precision (None: the configured default) round to bf16."""
    if precision is None:
        precision = jax.config.jax_default_matmul_precision
    return str(precision).upper() == "BF16_BF16_F32"


def bf16_weight_keys(spec: FusedSpec) -> tuple:
    """Weights the sequence path's kernels read as bf16 copies: bit-identical results
    at half the weight reads when every use is a bf16 tensor-core dot."""
    if not _BF16_STORAGE or spec.interpret or not _bf16_dots(spec.precision):
        return ()
    ok = lambda *dims: min(dims) >= MIN_LANES
    dims = {}
    if spec.pool_size:  # (member x slot) . (slot x lanes), and its transpose
        dims["ws"] = (spec.gp, spec.pp)
    if spec.mlp_depth == 1:  # hidden rows per chunk with branches / memory units
        hc = hidden_chunk(spec)
        dims.update(w1b=(hc, spec.nbp), w1m=(hc, spec.mp), w2=(hc, spec.mp))
    else:  # memory rows (per chunk) with branches / memory units
        rows = depth0_chunk(spec) or spec.mp
        dims.update(w2b=(rows, spec.nbp), w2m=(rows, spec.mp))
    return tuple(k for k, d in dims.items() if ok(*d))


def bf16_history_keys(spec: FusedSpec) -> tuple:
    """Histories the sequence path keeps as bf16 where bit-identical."""
    w = set(bf16_weight_keys(spec))
    keys = ["uhist"] if "ws" in w else []
    branches_ok = spec.mlp_depth == 1 and "w1b" in w and not spec.rec_branch
    if spec.pool_size and branches_ok:
        keys.append("bsum")
    if spec.fast_backward:
        if "ws" in w:
            keys.append("dbhist")
        if branches_ok:
            keys.append("bhist")
        if spec.mlp_depth == 1:
            if "w2" in w:
                keys += ["hhist", "dmohist"]
            if {"w1b", "w1m"} <= w:
                keys.append("dhhist")
    return tuple(keys)


def _pool_tables(spec: FusedSpec, conn, order) -> dict:
    """Pooled wiring: per shared list (branch k, group r; index k * runs + r) its
    member neurons (-1: none) and sources, plus the inverse wiring over the list
    entries (list, slot), whose input gradients the list kernel writes per step."""
    n, nb, p = spec.num_neuron, spec.num_branch, spec.num_syn
    g, gp, runs = spec.pool_size, spec.gp, spec.num_runs
    groups = jnp.pad(
        order.astype(jnp.int32), ((0, 0), (0, runs * g - n)), constant_values=-1
    )
    groups = groups.reshape(-1, runs, g)
    if gp <= g:
        groups = groups[..., :gp]
    else:
        groups = jnp.pad(groups, ((0, 0), (0, 0), (0, gp - g)), constant_values=-1)
    members = groups  # (branch, group, gp): branch pools, one block per branch
    branch = jnp.arange(nb)[:, None]
    src = conn.reshape(n, nb, p)[members[..., 0], branch]  # each group copies its first
    flat = src.reshape(-1)
    by_slot = jnp.argsort(flat, stable=True).astype(jnp.int32)
    counts = jnp.bincount(flat, length=spec.num_slots).astype(jnp.int32)
    return {
        "pool_neuron": members.reshape(-1),
        "pool_src": flat,
        "pool_inv_start": (jnp.cumsum(counts) - counts).astype(jnp.int32),
        "pool_inv_count": counts,
        "pool_inv_nb": (by_slot // p * spec.pp + by_slot % p).astype(jnp.int32),
    }


########## STEP MATH (USED BY THE SEQUENCE KERNELS) ##########


def lane_count(spec: FusedSpec, num_batch: int) -> int:
    """Batch lanes of the kernels (padded, batch innermost)."""
    del spec  # same for every layer
    return max(MIN_LANES, next_pow2(num_batch))


def group_lanes(spec: FusedSpec, num_batch: int) -> int:
    """Batch lanes per program: all (padded) samples, at most max_lanes."""
    return min(lane_count(spec, num_batch), spec.max_lanes)


def batch_groups(spec: FusedSpec, num_batch: int) -> int:
    """Sample groups, i.e. programs per neuron (grid axis 1 when more than one)."""
    return lane_count(spec, num_batch) // group_lanes(spec, num_batch)


def synapse_chunk(spec: FusedSpec) -> int:
    """Synapse slots per branch gathered at once (random wiring)."""
    return min(spec.pp, spec.max_synapse_chunk)


def synapse_chunk_count(spec: FusedSpec) -> int:
    """Chunks holding real synapse slots."""
    return -(-spec.num_syn // synapse_chunk(spec))


def hidden_chunk(spec: FusedSpec) -> int:
    """MLP hidden rows per piece."""
    return min(spec.hp, spec.mlp_chunk)


_MAX_UNROLLED_CHUNKS = 8  # more mlp chunks run as a loop (keeps kernels small)


def depth0_chunk(spec: FusedSpec) -> int:
    """Memory rows per piece of a depth-0 mlp (no hidden layer), 0 when unchunked."""
    if spec.mlp_depth != 0 or spec.mp <= spec.mlp_chunk:
        return 0
    return spec.mlp_chunk


def mlp_chunks(spec: FusedSpec) -> tuple:
    """The mlp's (first row, rows) chunks holding real rows."""
    if spec.mlp_depth == 1:
        rows, count = hidden_chunk(spec), spec.mlp_hidden
    elif depth0_chunk(spec):
        rows, count = depth0_chunk(spec), spec.num_memory
    else:
        return ((0, spec.mp),)
    return tuple((h0, rows) for h0 in range(0, -(-count // rows) * rows, rows))


def deferred_grad_keys(spec: FusedSpec) -> tuple:
    """fast_backward: the weights whose gradients are summed per window of steps."""
    if not spec.fast_backward:
        return ()
    return ("ws",) + (("w1b", "w1m", "w2") if spec.mlp_depth == 1 else ("w2b", "w2m"))


########## BATCH SPLIT ACROSS DEVICES, LAYER CACHE ##########


@functools.lru_cache(maxsize=None)
def _device_mesh(count: int):
    return jax.make_mesh((count,), ("batch",), axis_types=(AxisType.Auto,))


def data_split(axis_size: int):
    """Where to split a vmapped batch of axis_size samples across devices."""
    ctx = jax.sharding.get_abstract_mesh()
    if not ctx.empty:
        names = ctx.axis_names
        if (
            "batch" not in names
            or ctx.axis_types[names.index("batch")] == AxisType.Manual
        ):
            return None
        mesh, count = ctx, ctx.shape["batch"]
    else:
        count = jax.device_count()
        mesh = _device_mesh(count) if count > 1 else None
    if count <= 1 or axis_size % count:
        return None
    return mesh, "batch", count


def batch_specs(batched, axis: str):
    """Leaves with a leading batch axis split over axis, the rest replicated."""
    return jtu.tree_map(lambda b: P(axis) if b else P(), batched)


def split_batch(fun, split, in_specs, out_specs):
    """fun on each device's share of the batch (shard_map over the batch axis)."""
    mesh, axis, _ = split
    return jax.shard_map(
        fun, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
        axis_names={axis}, check_vma=False,
    )  # fmt: skip
