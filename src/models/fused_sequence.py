"""The fused path of ELMNetwork (opt-in: ELMNetwork(fused=True)): the whole time loop.

One custom_vjp covers the whole sequence (so XLA saves no per-step residuals):
  - forward: a fori_loop over time; each layer's kernel reads its previous state
    from history buffers and writes the new state into them in place;
  - backward: a reverse fori_loop; the kernels accumulate weight gradients and
    carry the state cotangents in place.
Per time step that leaves one forward kernel per layer and two backward kernels
per layer (plus the list kernels of pooled layers). The kernels are in
triton_kernels.py, launched by triton_launch.launch.
"""

import dataclasses
import functools

import jax
import jax.lax as jlx
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.custom_batching import custom_vmap
from jax.sharding import PartitionSpec as P

from . import fused_layer as fk
from . import triton_kernels as tk
from .triton_launch import launch

F32 = fk.F32
# ensemble recording name -> internal buffer name
RECORDINGS = {
    "output": "output",
    "preact": "preact",
    "readout": "readout",
    "mlp": "mlp",
    "branch": "branch",
    "memory": "memory",
}


@dataclasses.dataclass(frozen=True)
class SeqSpec:
    layers: tuple  # FusedSpec per layer
    seq_len: int
    rec_memory: bool  # "memory" recording requested
    dropout: bool = False  # variational dropout masks in init ("din", "drec")


def check_fused_network_support(net) -> None:
    """Asserts that the network and every layer only use settings the kernels implement."""
    for layer in net.layers:
        fk.check_fused_support(layer)


########## LAYOUT HELPERS ##########


def _to_lanes(x, num_batch: int, lanes: int, feat_pad: int = 0):
    """(batch or 1, ...) -> (..., lanes): batch innermost, zero padded lanes."""
    x = jnp.broadcast_to(x, (num_batch,) + x.shape[1:])
    if feat_pad > x.shape[-1]:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, feat_pad - x.shape[-1])])
    x = jnp.moveaxis(x, 0, -1)
    if lanes > num_batch:
        x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, lanes - num_batch)])
    return x


def _from_lanes(x, num_batch: int, feat: int = 0):
    """(..., lanes) -> (batch, ...), feature dim cut to feat."""
    x = jnp.moveaxis(x[..., :num_batch], -1, 0)
    return x[..., :feat] if feat else x


# fast_backward: the per-step inputs of the deferred mlp weight gradients
RECEIPTS = {"dmo": "dmohist", "h": "hhist", "dhpre": "dhhist"}


def _hist_dtype(spec, key):
    """A fast_backward history's dtype: bf16 where that is bit-identical."""
    return jnp.bfloat16 if key in fk.bf16_history_keys(spec) else F32


# Within-step indices (one time slot of a buffer, or a per-step buffer) are int32.
_MAX_FLAT = 2**31 - 1


def _max_step_size(sspec, lanes: int) -> int:
    """Elements of a call's largest time slot: (row width) x lanes."""
    size = 0
    for spec in sspec.layers:
        width = max(spec.num_slots, spec.num_neuron * spec.mp)
        if spec.rec_branch:
            width = max(width, spec.num_neuron * spec.nbp)
        if spec.fast_backward:  # branch sums (plus a dump row), mlp hidden rows
            width = max(
                width, (spec.num_neuron + 1) * spec.nbp, spec.num_neuron * spec.hp
            )
        size = max(size, width * lanes)
    return size


def _check_flat_size(sspec, lanes: int) -> None:
    size = _max_step_size(sspec, lanes)
    if size > _MAX_FLAT:
        raise ValueError(
            f"fused: one time step with {lanes} sample lanes needs "
            f"{size:,} elements, beyond 32-bit indexing; use smaller batches"
        )


def _seq_lanes(sspec, num_batch: int) -> int:
    return fk.lane_count(sspec.layers[0], num_batch)  # same for every layer


########## KERNELS ##########


def _list_grid(spec, num_batch):
    """The list kernels' grid: (shared list[, sample group])."""
    groups = fk.batch_groups(spec, num_batch)
    return (spec.num_lists,) if groups == 1 else (spec.num_lists, groups)


def _lanes(spec, num_batch, lanes) -> tk.Lanes:
    """The batch as a kernel sees it: lanes per program, of the whole padded batch."""
    return tk.Lanes(lanes=fk.group_lanes(spec, num_batch), stride=lanes, num_batch=num_batch,
                    grouped=fk.batch_groups(spec, num_batch) > 1)  # fmt: skip


def _lists(spec) -> tk.Lists:
    """The compile-time description of a pooled layer's shared lists."""
    return tk.Lists(
        num_input=spec.num_input, num_neuron=spec.num_neuron, num_branch=spec.num_branch,
        nbp=spec.nbp, num_syn=spec.num_syn, pp=spec.pp, gp=spec.gp, num_runs=spec.num_runs,
        gain=float(spec.gain), window=spec.grad_window,
    )


def dot_mode(spec) -> str:
    """The Triton dot precision for this layer, following jax_default_matmul_precision."""
    precision = spec.precision or jax.config.jax_default_matmul_precision
    name = str(precision).lower() if precision is not None else "default"
    if spec.interpret:
        return "ieee"
    modes = {"default": "tf32", "bfloat16": "tf32", "high": "tf32", "tensorfloat32": "tf32",
             "highest": "ieee", "float32": "ieee", "f32_f32_f32": "ieee",
             "tf32_tf32_f32": "tf32", "tf32_tf32_f32_x3": "tf32x3", "bf16_bf16_f32": "bf16"}  # fmt: skip
    if name not in modes:
        raise NotImplementedError(f"fused kernels: unsupported matmul precision {precision!r}")
    return modes[name]


def _pool_bsum(spec, num_batch, lanes, t_arr, uhist, fp, bhist=None):
    """Branch sums at step t from the list kernel, (neuron, padded branch, lanes);
    with bhist (fast_backward): written into its step t instead, in place."""
    size = spec.num_neuron * spec.nbp * lanes  # + one dump row (see list_members)
    inplace = [bhist] if bhist is not None else []
    outputs = [] if bhist is not None else [
        jax.ShapeDtypeStruct((size + lanes,), _hist_dtype(spec, "bsum"))]  # fmt: skip
    updated, fresh = launch(
        tk.elm_seq_pool_fwd, name="elm_seq_pool_fwd",
        grid=_list_grid(spec, num_batch),
        inputs=[t_arr, uhist, fp["pool_src"], fp["pool_neuron"], fp["ws"].reshape(-1)],
        inplace=inplace, outputs=outputs, interpret=spec.interpret, num_warps=spec.num_warps,
        constants=dict(P=_lists(spec), B=_lanes(spec, num_batch, lanes), DOT=dot_mode(spec),
                       HISTORY=bhist is not None),
    )  # fmt: skip
    if bhist is not None:
        return updated[0]
    return fresh[0][:size].reshape(spec.num_neuron, spec.nbp, lanes)


########## FORWARD ##########


def _layer_bufs(spec, seq_len, lanes, num_batch, state0, xhist=None, drop=None):
    """History / recording buffers of one layer (flat), slot 0 = initial state."""
    nn, s_tot, i_n = spec.num_neuron, spec.num_slots, spec.num_input
    t1 = seq_len + 1
    a0 = _to_lanes(state0["a"], num_batch, lanes)
    m0 = _to_lanes(state0["m"], num_batch, lanes, spec.mp)
    u = jnp.zeros((t1, s_tot, lanes), F32)
    if xhist is not None:
        u = u.at[:seq_len, : xhist.shape[1]].set(xhist)
    if drop is None:
        u = u.at[0, i_n:].set(a0)
    else:
        u = u.at[0, i_n:].set(a0 * drop["drec"])
    bufs = {
        "ahist": jnp.zeros((t1, nn, lanes), F32).at[0].set(a0).reshape(t1, -1),
        "mhist": jnp.zeros((t1, nn, spec.mp, lanes), F32).at[0].set(m0).reshape(t1, -1),
        "uhist": u.reshape(t1, -1).astype(_hist_dtype(spec, "uhist")),
    }
    if spec.fast_backward:  # the list kernel's branch sums (plus its dump row)
        shape = (seq_len, (nn * spec.nbp + 1) * lanes)
        bufs["bhist"] = jnp.zeros(shape, _hist_dtype(spec, "bhist"))
    if spec.high_pass:
        r0 = _to_lanes(state0["r"][..., 0], num_batch, lanes)
        bufs["rhist"] = jnp.zeros((t1, nn, lanes), F32).at[0].set(r0).reshape(t1, -1)
    shapes = {
        "preact": (seq_len, nn, lanes),
        "readout": (seq_len, nn, lanes),
        "mlp": (seq_len, nn, spec.mp, lanes),
        "branch": (seq_len, nn, spec.nbp, lanes),
    }
    for flag, key in (
        (spec.rec_preact, "preact"),
        (spec.rec_readout, "readout"),
        (spec.rec_mlp, "mlp"),
        (spec.rec_branch, "branch"),
    ):
        if flag:
            bufs[key] = jnp.zeros(shapes[key], F32).reshape(seq_len, -1)
    return bufs


def _fwd_step(
    spec,
    num_batch,
    lanes,
    t_arr,
    bufs,
    unext,
    s_next,
    fp,
    drop=None,
    dnext=None,
):
    """One step of a layer's neurons (elm_seq_fwd), after the list kernel's branch sums."""
    bufs = dict(bufs)
    bsum = None  # branch sums from the list kernel (pooled), else gathered in the kernel
    if spec.fast_backward:  # into step t of their history, which the backward reads again
        bufs["bhist"] = _pool_bsum(
            spec, num_batch, lanes, t_arr, bufs["uhist"], fp, bufs["bhist"]
        )
        bsum = bufs["bhist"]
    elif spec.pool_size:
        bsum = _pool_bsum(spec, num_batch, lanes, t_arr, bufs["uhist"], fp).reshape(-1)
    recordings = ("preact", "readout", "mlp", "branch")
    groups = fk.batch_groups(spec, num_batch)
    (uhist, ahist, mhist, rhist, unext_new, *recorded), _ = launch(
        tk.elm_seq_fwd, name="elm_seq_fwd",
        grid=(spec.num_neuron,) if groups == 1 else (spec.num_neuron, groups),
        inputs=[t_arr,
                None if drop is None else drop["drec"].reshape(-1),
                None if (drop is None or unext is None) else dnext["din"].reshape(-1),
                bsum, _neuron_weights(spec, fp)],
        inplace=[bufs["uhist"], bufs["ahist"], bufs["mhist"], bufs.get("rhist"), unext,
                 *(bufs.get(k) for k in recordings)],
        interpret=spec.interpret, num_warps=spec.num_warps,
        constants=dict(
            L=_neurons(spec), B=_lanes(spec, num_batch, lanes), DOT=dot_mode(spec),
            NEXT_WIDTH=s_next,
        ),
    )  # fmt: skip
    out = dict(bufs, uhist=uhist, ahist=ahist, mhist=mhist)
    if rhist is not None:
        out["rhist"] = rhist
    if unext is not None:
        out["unext"] = unext_new
    out.update({k: v for k, v in zip(recordings, recorded) if v is not None})
    return out


def _neuron_weights(spec, fp) -> tuple:
    """The neuron kernels' weights, as one tuple in the kernels' order."""
    used = lambda k: k in fp and not (spec.pool_size and k in ("conn", "ws"))
    return tuple(fp[k] if used(k) else None
                 for k in ("conn", "ws", "w1b", "w1m", "b1", "w2", "w2b", "w2m", "b2", "wy",
                           "km", "kl", "bias"))  # fmt: skip


def _neurons(spec) -> tk.Neurons:
    """The compile-time description of a layer's neurons: sizes, math and pieces."""
    if spec.mlp_depth == 1:
        rows = fk.hidden_chunk(spec)
        pieces = -(-spec.mlp_hidden // rows)
    else:
        rows = fk.depth0_chunk(spec)
        pieces = -(-spec.num_memory // rows) if rows else 0
    activation = {"relu": tk.ACT_RELU, "spike": tk.ACT_SPIKE, "identity": tk.ACT_IDENTITY}
    mlp_act = {"relu": tk.MLP_ACT_RELU, "relu_squared": tk.MLP_ACT_RELU_SQ, "silu": tk.MLP_ACT_SILU}
    return tk.Neurons(
        num_input=spec.num_input, num_neuron=spec.num_neuron, num_branch=spec.num_branch,
        nbp=spec.nbp, num_syn=spec.num_syn, num_memory=spec.num_memory, mp=spec.mp,
        num_hidden=spec.mlp_hidden if spec.mlp_depth == 1 else 1, hp=spec.hp,
        depth=spec.mlp_depth, mlp_act=mlp_act[spec.mlp_act],
        activation=activation[spec.activation],
        threshold=float(spec.spike_threshold), scale=float(spec.spike_scale),
        kappa_r=float(spec.kappa_r), gain=float(spec.gain), hc=max(rows, 1), pieces=pieces,
        unroll=pieces if 1 <= pieces <= fk._MAX_UNROLLED_CHUNKS else 1,
        syn_chunk=fk.synapse_chunk(spec), syn_chunks=fk.synapse_chunk_count(spec),
        window=spec.grad_window,
    )


def _seq_forward(sspec: SeqSpec, num_batch, lanes, xhist, init_int, fps, drops=None):
    seq_len, layers = sspec.seq_len, sspec.layers
    drop = lambda l: None if drops is None else drops[l]
    bufs0 = tuple(
        _layer_bufs(
            spec,
            seq_len,
            lanes,
            num_batch,
            init_int[l],
            xhist if l == 0 else None,
            drop(l),
        )
        for l, spec in enumerate(layers)
    )

    def body(t, bufs):
        t_arr = jnp.reshape(t, (1,)).astype(jnp.int32)
        bufs = [dict(b) for b in bufs]
        for l, spec in enumerate(layers):
            last = l == len(layers) - 1
            unext = None if last else bufs[l + 1]["uhist"]
            s_next = 0 if last else layers[l + 1].num_slots
            res = _fwd_step(
                spec, num_batch, lanes, t_arr, bufs[l], unext, s_next,
                fps[l], drop(l), None if last else drop(l + 1),
            )
            if not last:
                bufs[l + 1]["uhist"] = res.pop("unext")
            bufs[l] = res
        return tuple(bufs)

    return jlx.fori_loop(0, seq_len, body, bufs0)


def _drop_lanes(sspec: SeqSpec, init, num_batch: int, lanes: int):
    """Dropout masks per layer in lanes layout ((input or neuron, lanes)), or None."""
    if not sspec.dropout:
        return None
    to = lambda x: _to_lanes(x, num_batch, lanes)
    return tuple({"din": to(i["din"]), "drec": to(i["drec"])} for i in init)


def _kernel_fps(sspec: SeqSpec, fps):
    """The weights as the kernels read them: bf16 copies where bit-identical."""
    out = []
    for spec, fp in zip(sspec.layers, fps):
        keys = fk.bf16_weight_keys(spec)
        out.append(
            {k: v.astype(jnp.bfloat16) if k in keys else v for k, v in fp.items()}
        )
    return tuple(out)


def _seq_fwd_impl(sspec: SeqSpec, num_batch: int, xs, init, fps):
    """Batched forward; act inputs carry a leading dim (num_batch or 1)."""
    fps = _kernel_fps(sspec, fps)
    lanes = _seq_lanes(sspec, num_batch)
    _check_flat_size(sspec, lanes)
    seq_len = sspec.seq_len
    xhist = _to_lanes(xs, num_batch, lanes)  # (time, input, lanes)
    drops = _drop_lanes(sspec, init, num_batch, lanes)
    if drops is not None:  # masked first-layer inputs
        xhist = xhist * drops[0]["din"][None]
    bufs = _seq_forward(sspec, num_batch, lanes, xhist, init, fps, drops)
    final, recs, hist = [], [], []
    for l, spec in enumerate(sspec.layers):
        nn, nmem = spec.num_neuron, spec.num_memory
        b = bufs[l]
        ahist = b["ahist"].reshape(seq_len + 1, nn, lanes)
        mhist = b["mhist"].reshape(seq_len + 1, nn, spec.mp, lanes)
        fin = {
            "a": _from_lanes(ahist[seq_len], num_batch),
            "m": _from_lanes(mhist[seq_len], num_batch, nmem),
        }
        act_hist = _from_lanes(ahist[1:], num_batch)
        rec_out = {"activity": act_hist}
        if spec.high_pass:
            rhist = b["rhist"].reshape(seq_len + 1, nn, lanes)
            fin["r"] = _from_lanes(rhist[seq_len], num_batch)[..., None]
        if spec.rec_output:
            rec_out["output"] = act_hist[..., None]
        if spec.rec_preact:
            preact = b["preact"].reshape(seq_len, nn, lanes)
            rec_out["preact"] = _from_lanes(preact, num_batch)[..., None]
        if spec.rec_readout:
            readout = b["readout"].reshape(seq_len, nn, lanes)
            rec_out["readout"] = _from_lanes(readout, num_batch)[..., None]
        if spec.rec_mlp:
            mlp = b["mlp"].reshape(seq_len, nn, spec.mp, lanes)
            rec_out["mlp"] = _from_lanes(mlp, num_batch, nmem)
        if spec.rec_branch:
            br = b["branch"].reshape(seq_len, nn, spec.nbp, lanes)
            rec_out["branch"] = _from_lanes(br, num_batch, spec.num_branch)
        if sspec.rec_memory:
            rec_out["memory"] = _from_lanes(mhist[1:], num_batch, nmem)
        final.append(fin)
        recs.append(rec_out)
        h = {"ahist": b["ahist"], "mhist": b["mhist"], "uhist": b["uhist"]}
        if spec.fast_backward:
            h["bhist"] = b["bhist"]
        if spec.high_pass:
            h["rhist"] = b["rhist"]
        hist.append(h)
    hist = {
        "layers": tuple(hist),
        "batch": jnp.zeros((num_batch,), F32),  # batch size of the buffers (shape)
    }
    return tuple(final), tuple(recs), hist


########## BACKWARD ##########


def _bwd_step(spec, num_batch, lanes, t_arr, h, cts, carry, dxbuf, fp):
    """One backward step of a layer's neurons (elm_seq_bwd), then, pooled, of its lists."""
    bsum = None
    if spec.fast_backward:  # kept by the forward
        bsum = h["bhist"]
    elif spec.pool_size:  # recomputed
        bsum = _pool_bsum(spec, num_batch, lanes, t_arr, h["uhist"], fp).reshape(-1)
    deferred = fk.deferred_grad_keys(spec)
    keys = tuple(
        k for k in spec.grad_keys()
        if not (spec.pool_size and k == "ws") and k not in deferred
    )  # fmt: skip
    acc_keys = ("ws", "w1b", "w1m", "b1", "w2", "w2b", "w2m", "b2", "wy", "km", "kl", "bias")
    kept_keys = ("dbhist",) + tuple(RECEIPTS.values())
    ct_keys = ("preact", "readout", "mlp", "branch", "memory")
    groups = fk.batch_groups(spec, num_batch)
    B = _lanes(spec, num_batch, lanes)
    (dmcarry, drcarry, kept, acc), (dbs,) = launch(
        tk.elm_seq_bwd, name="elm_seq_bwd",
        grid=(spec.num_neuron,) if groups == 1 else (spec.num_neuron, groups),
        inputs=[t_arr, h["uhist"], h["mhist"], h.get("rhist"), bsum,
                cts["cact"], carry["dacarry"], dxbuf,
                *(cts.get("c" + k) for k in ct_keys), _neuron_weights(spec, fp)],
        inplace=[carry["dmcarry"], carry.get("drcarry"),
                 tuple(carry.get(k) for k in kept_keys),
                 tuple(carry["acc"][k] if k in keys else None for k in acc_keys)],
        outputs=[jax.ShapeDtypeStruct(
            (1,) if spec.fast_backward else (spec.num_neuron * spec.nbp * lanes,), F32)],
        interpret=spec.interpret, num_warps=spec.num_warps,
        constants=dict(L=_neurons(spec), B=B, DOT=dot_mode(spec)),
    )  # fmt: skip
    new = dict(carry, dmcarry=dmcarry)
    if spec.high_pass:
        new["drcarry"] = drcarry
    new.update({k: v for k, v in zip(kept_keys, kept) if v is not None})
    new["acc"] = dict(carry["acc"])
    new["acc"].update({k: v for k, v in zip(acc_keys, acc) if v is not None})
    if not spec.pool_size:
        return new, dbs
    grads = new["dbhist"] if spec.fast_backward else dbs
    window = spec.fast_backward
    (acc_ws,), (dlist,) = launch(
        tk.elm_seq_pool_bwd, name="elm_seq_pool_bwd", grid=_list_grid(spec, num_batch),
        inputs=[t_arr, None if window else h["uhist"], fp["pool_src"], fp["pool_neuron"],
                fp["ws"].reshape(-1), grads],
        inplace=[None if window else carry["acc"]["ws"]],
        outputs=[jax.ShapeDtypeStruct((spec.num_lists * spec.pp * lanes,), F32)],
        interpret=spec.interpret, num_warps=spec.num_warps,
        constants=dict(P=_lists(spec), B=B, DOT=dot_mode(spec)),
    )  # fmt: skip
    if not window:
        new["acc"]["ws"] = acc_ws
    return new, dlist


def _deferred_grads(spec, num_batch, h, carry, fp, t0):
    """fast_backward: the synapse and mlp weight gradients of the window starting at step t0."""
    acc = dict(carry["acc"])
    lanes = h["uhist"].shape[1] // spec.num_slots
    B = _lanes(spec, num_batch, lanes)
    (acc["ws"],), _ = launch(
        tk.elm_seq_pool_wgrad, name="elm_seq_pool_wgrad", grid=_list_grid(spec, num_batch),
        inputs=[t0, h["uhist"], fp["pool_src"], fp["pool_neuron"], carry["dbhist"]],
        inplace=[acc["ws"]], interpret=spec.interpret, num_warps=spec.num_warps,
        constants=dict(P=_lists(spec), B=B, DOT=dot_mode(spec)),
    )  # fmt: skip
    keys = ("ws", "w1b", "w1m", "b1", "w2", "w2b", "w2m", "b2", "wy", "km", "kl", "bias")
    deferred = set(fk.deferred_grad_keys(spec)) - {"ws"}
    groups = fk.batch_groups(spec, num_batch)
    (updated,), _ = launch(
        tk.elm_seq_mlp_wgrad, name="elm_seq_mlp_wgrad",
        grid=(spec.num_neuron,) if groups == 1 else (spec.num_neuron, groups),
        inputs=[t0, h["bhist"], h["mhist"],
                tuple(carry.get(k) for k in ("dbhist",) + tuple(RECEIPTS.values())),
                _neuron_weights(spec, fp)],
        inplace=[tuple(acc[k] if k in deferred else None for k in keys)],
        interpret=spec.interpret, num_warps=spec.num_warps,
        constants=dict(L=_neurons(spec), B=B, DOT=dot_mode(spec)),
    )  # fmt: skip
    acc.update({k: v for k, v in zip(keys, updated) if v is not None})
    return {**carry, "acc": acc}


def _ingrad_step(spec, num_batch, lanes, t_arr, dbs, fp, dx, carry, layer0, drop=None):
    inv = "pool_inv_" if spec.pool_size else "inv_"
    (dx, dacarry), _ = launch(
        tk.elm_seq_ingrad, name="elm_seq_ingrad", grid=(spec.num_slots,),
        inputs=[
            t_arr, fp[inv + "start"], fp[inv + "count"], fp[inv + "nb"], dbs,
            None if spec.pool_size else fp["inv_w"],
            None if drop is None else drop["din"].reshape(-1),
            None if drop is None else drop["drec"].reshape(-1),
        ],
        inplace=[dx, carry["dacarry"]],
        interpret=spec.interpret, num_warps=spec.num_warps,
        constants=dict(
            NUM_INPUT=spec.num_input, NUM_NEURON=spec.num_neuron, LANES=lanes,
            CHUNK=max(16, fk.INGRAD_TILE // lanes), DX_HISTORY=layer0,
        ),
    )  # fmt: skip
    return {**carry, "dacarry": dacarry}, dx


def _seq_bwd_impl(sspec, num_batch, flags, xs, init, fps, hist, ct_final, ct_recs):
    """Batched backward; returns cotangents (batch-summed for unbatched inputs)."""
    fps = _kernel_fps(sspec, fps)
    lanes = _seq_lanes(sspec, num_batch)
    _check_flat_size(sspec, lanes)
    seq_len = sspec.seq_len
    layers = sspec.layers
    xs_flag, init_flags = flags
    drops = _drop_lanes(sspec, init, num_batch, lanes)

    def lanes_flat(x, feat_pad=0):
        return _to_lanes(x, num_batch, lanes, feat_pad).reshape(-1)

    def lanes_hist(x, feat_pad=0):
        return _to_lanes(x, num_batch, lanes, feat_pad).reshape(seq_len, -1)

    def acc_shape(l, key):
        groups = fk.batch_groups(layers[l], num_batch)
        shape = fps[l][key].shape
        return shape if groups == 1 else (groups,) + shape

    carries, cts, dxbufs = [], [], []
    for l, spec in enumerate(layers):
        nn = spec.num_neuron
        c = {
            "dacarry": lanes_flat(ct_final[l]["a"]),
            "dmcarry": lanes_flat(ct_final[l]["m"], spec.mp),
            "acc": {k: jnp.zeros(acc_shape(l, k), F32) for k in spec.grad_keys()},
        }
        if spec.pool_size:  # a row per sample group, plus the list kernel's dump slot
            groups = fk.batch_groups(spec, num_batch)
            size = spec.num_neuron * spec.num_synapse
            c["acc"]["ws"] = jnp.zeros((groups, size + spec.pp), F32)
        if spec.high_pass:
            c["drcarry"] = lanes_flat(ct_final[l]["r"][..., 0])
        if spec.fast_backward:
            rows = {"dbhist": spec.nbp, "dmohist": spec.mp}
            if spec.mlp_depth == 1:
                rows.update(hhist=spec.hp, dhhist=spec.hp)
            if seq_len % spec.grad_window:
                raise ValueError(
                    f"fast_backward: the window ({spec.grad_window}) must divide the "
                    f"sequence length ({seq_len})"
                )
            for k, width in rows.items():
                shape = (spec.grad_window, nn * width * lanes)
                c[k] = jnp.zeros(shape, _hist_dtype(spec, k))
        carries.append(c)
        cact = ct_recs[l]["activity"]
        if "output" in ct_recs[l]:  # output recording equals activity[..., None]
            cact = cact + ct_recs[l]["output"][..., 0]
        ct_l = {"cact": lanes_hist(cact)}
        for name in ("preact", "readout", "mlp", "branch", "memory"):
            if name in ct_recs[l]:
                x = ct_recs[l][name]
                if name in ("preact", "readout"):
                    x = x[..., 0]
                pad = {"mlp": spec.mp, "memory": spec.mp, "branch": spec.nbp}.get(
                    name, 0
                )
                ct_l["c" + name] = lanes_hist(x, pad)
        cts.append(ct_l)
        dxbufs.append(jnp.zeros((nn * lanes,), F32) if l < len(layers) - 1 else None)
    dxhist = jnp.zeros((seq_len, layers[0].num_input * lanes), F32)

    def body(i, state):
        carries, dxbufs, dxhist = state
        carries, dxbufs = list(carries), list(dxbufs)
        t = seq_len - 1 - i
        t_arr = jnp.reshape(t, (1,)).astype(jnp.int32)
        for l in reversed(range(len(layers))):
            spec = layers[l]
            h = hist["layers"][l]
            carries[l], dbs = _bwd_step(
                spec, num_batch, lanes, t_arr, h, cts[l], carries[l], dxbufs[l], fps[l]
            )
            dx = dxhist if l == 0 else dxbufs[l - 1]
            carries[l], dx = _ingrad_step(
                spec, num_batch, lanes, t_arr, dbs, fps[l], dx, carries[l],
                l == 0, None if drops is None else drops[l],
            )
            if l == 0:
                dxhist = dx
            else:
                dxbufs[l - 1] = dx
        for l, spec in enumerate(layers):  # a window's first step: its weight gradients
            if spec.fast_backward:
                flush = functools.partial(
                    _deferred_grads, spec, num_batch, hist["layers"][l],
                    fp=fps[l], t0=t_arr,
                )
                carries[l] = jlx.cond(
                    t % spec.grad_window == 0, flush, lambda c: c, carries[l]
                )
        return tuple(carries), tuple(dxbufs), dxhist

    carries, _, dxhist = jlx.fori_loop(
        0, seq_len, body, (tuple(carries), tuple(dxbufs), dxhist)
    )

    def reduce(x, batched):
        return x if batched else jnp.sum(x, axis=0, keepdims=True)

    d_xs = _from_lanes(dxhist.reshape(seq_len, layers[0].num_input, lanes), num_batch)
    d_xs = reduce(d_xs, xs_flag)
    d_init, d_fps = [], []
    for l, spec in enumerate(layers):
        nn, c = spec.num_neuron, carries[l]
        d = {
            "a": reduce(
                _from_lanes(c["dacarry"].reshape(nn, lanes), num_batch),
                init_flags[l]["a"],
            ),
            "m": reduce(
                _from_lanes(
                    c["dmcarry"].reshape(nn, spec.mp, lanes), num_batch, spec.num_memory
                ),
                init_flags[l]["m"],
            ),
        }
        if spec.high_pass:
            r = _from_lanes(c["drcarry"].reshape(nn, lanes), num_batch)[..., None]
            d["r"] = reduce(r, init_flags[l]["r"])
        if sspec.dropout:
            d["din"] = jnp.zeros_like(init[l]["din"])
            d["drec"] = jnp.zeros_like(init[l]["drec"])
        d_init.append(d)
        if spec.pool_size:  # drop the dump slot
            size = spec.num_neuron * spec.num_synapse
            c["acc"]["ws"] = c["acc"]["ws"][:, :size].reshape(acc_shape(l, "ws"))
        if fk.batch_groups(layers[l], num_batch) > 1:  # sum the sample groups
            c["acc"] = {k: jnp.sum(v, axis=0) for k, v in c["acc"].items()}
        d_fps.append(c["acc"])
    return d_xs, tuple(d_init), tuple(d_fps)


########## AUTODIFF AND BATCHING GLUE ##########


def _true(tree):
    return jtu.tree_map(lambda _: True, tree)


def _lead(tree, flags=None):
    if flags is None:
        return jtu.tree_map(lambda v: v[None], tree)
    return jtu.tree_map(lambda v, b: v if b else v[None], tree, flags)


def _unlead(tree):
    return jtu.tree_map(lambda v: v[0], tree)


@functools.lru_cache(maxsize=None)
def _seq_fwd_fn(sspec: SeqSpec):
    def unbatched(xs, init, fps):
        final, recs, hist = _seq_fwd_impl(sspec, 1, *_lead((xs, init)), fps)
        return _unlead(final), _unlead(recs), hist

    fun = custom_vmap(unbatched)

    @fun.def_vmap
    def rule(axis_size, in_batched, xs, init, fps):
        if any(jtu.tree_leaves(in_batched[2])):
            raise NotImplementedError("fused: batched (vmapped) weights")
        flags = tuple(in_batched[:2])
        lead = _lead((xs, init), flags)
        split = fk.data_split(axis_size)
        if split is None:
            final, recs, hist = _seq_fwd_impl(sspec, axis_size, *lead, fps)
        else:  # each device its share of the samples
            _, axis, count = split

            def local(lead, fps):
                final, recs, hist = _seq_fwd_impl(sspec, axis_size // count, *lead, fps)
                return final, recs, _lead(hist)

            in_specs = (fk.batch_specs(flags, axis), P())
            out_specs = (P(axis), P(axis), P(axis))
            final, recs, hist = fk.split_batch(local, split, in_specs, out_specs)(
                lead, fps
            )
        out_batched = (_true(final), _true(recs), jtu.tree_map(lambda _: False, hist))
        return (final, recs, hist), out_batched

    return fun


@functools.lru_cache(maxsize=None)
def _seq_bwd_fn(sspec: SeqSpec):
    def unbatched(xs, init, fps, hist, ct_final, ct_recs):
        if hist["batch"].ndim == 2:
            hist = _unlead(hist)
        num_batch = hist["batch"].shape[0]
        flags = jtu.tree_map(lambda _: False, (xs, init))
        d_xs, d_init, d_fps = _seq_bwd_impl(
            sspec,
            num_batch,
            flags,
            *_lead((xs, init)),
            fps,
            hist,
            *_lead((ct_final, ct_recs)),
        )
        return _unlead(d_xs), _unlead(d_init), d_fps

    fun = custom_vmap(unbatched)

    @fun.def_vmap
    def rule(axis_size, in_batched, xs, init, fps, hist, ct_final, ct_recs):
        xs_b, init_b, fps_b, hist_b, ctf_b, ctr_b = in_batched
        if any(jtu.tree_leaves(fps_b)) or any(jtu.tree_leaves(hist_b)):
            raise NotImplementedError("fused: batched weights or residuals")
        flags = (xs_b, init_b)
        lead = _lead((xs, init), flags)
        cts = _lead((ct_final, ct_recs), (ctf_b, ctr_b))
        split = fk.data_split(axis_size)
        if split is None:
            d_xs, d_init, d_fps = _seq_bwd_impl(
                sspec, axis_size, flags, *lead, fps, hist, *cts
            )
        else:  # each device its share; batch sums add up the other devices' shares
            _, axis, count = split

            def local(lead, fps, hist, cts):
                d_xs, d_init, d_fps = _seq_bwd_impl(
                    sspec, axis_size // count, flags, *lead, fps, _unlead(hist), *cts
                )
                add = lambda v, b: v if b else jax.lax.psum(v, axis)
                d_init = jtu.tree_map(add, d_init, init_b)
                d_fps = jax.lax.psum(d_fps, axis)
                return add(d_xs, xs_b), d_init, d_fps

            specs = fk.batch_specs(flags, axis)
            cspecs = fk.batch_specs((ctf_b, ctr_b), axis)
            in_specs = (specs, P(), P(axis), cspecs)
            out_specs = (*specs, P())
            d_xs, d_init, d_fps = fk.split_batch(
                local, split, in_specs, out_specs
            )(lead, fps, hist, cts)
        unl = lambda v, b: v if b else v[0]
        d_xs = unl(d_xs, xs_b)
        d_init = jtu.tree_map(unl, d_init, init_b)
        out_b = (xs_b, init_b, jtu.tree_map(lambda _: False, d_fps))
        return (d_xs, d_init, d_fps), out_b

    return fun


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _fused_sequence(sspec: SeqSpec, xs, init, fps):
    # without a backward (e.g. evaluation): no fast_backward histories, same results
    plain = tuple(dataclasses.replace(s, grad_window=1) for s in sspec.layers)
    final, recs, _ = _seq_fwd_fn(dataclasses.replace(sspec, layers=plain))(
        xs, init, fps
    )
    return final, recs


def _fused_sequence_fwd(sspec, xs, init, fps):
    final, recs, hist = _seq_fwd_fn(sspec)(xs, init, fps)
    return (final, recs), (xs, init, fps, hist)


def _fused_sequence_bwd(sspec, res, ct):
    xs, init, fps, hist = res
    ct_final, ct_recs = ct
    d_xs, d_init, d_fps = _seq_bwd_fn(sspec)(xs, init, fps, hist, ct_final, ct_recs)
    d_fps = tuple({k: d.get(k, None) for k in fp} for d, fp in zip(d_fps, fps))
    return d_xs, d_init, d_fps


_fused_sequence.defvjp(_fused_sequence_fwd, _fused_sequence_bwd)


########## NETWORK INTEGRATION ##########


def fused_network_sequence(net, carry, x, monitor, inference=True):
    """Fused replacement for the time scan of ELMNetwork.__call__ (per sample)."""
    layer_carries, key, const_key = carry
    _, ensemble_monitor = monitor if monitor else ([], [])
    specs = tuple(fk.layer_fused_spec(layer, ensemble_monitor) for layer in net.layers)
    dropout = not inference and any(
        layer.input_dropout > 0.0 or layer.recurrent_dropout > 0.0
        for layer in net.layers
    )
    sspec = SeqSpec(
        layers=specs,
        seq_len=int(x.shape[0]),
        rec_memory="memory" in ensemble_monitor,
        dropout=dropout,
    )
    init, fps = [], []
    for l, layer in enumerate(net.layers):
        (state, _, layer_key), (ens_state, _, _) = layer_carries[l]
        s = {"a": state["a_t"], "m": ens_state["m_t"]}
        if dropout:  # variational masks, drawn exactly as in ELMLayer.dynamics
            input_key, recurrent_key = jrandom.split(layer_key, 2)
            ones = lambda n: jnp.ones((n,), F32)
            s["din"] = layer.apply_activity_dropout(
                ones(specs[l].num_input), layer.input_dropout, False, input_key
            )
            s["drec"] = layer.apply_activity_dropout(
                ones(specs[l].num_neuron), layer.recurrent_dropout, False, recurrent_key
            )
        if specs[l].high_pass:
            s["r"] = ens_state["r_t"]
        init.append(s)
        fps.append(fk.prepare_fused_params(specs[l], fk.layer_step_params(layer)))
    final, recs = _fused_sequence(sspec, x.astype(F32), tuple(init), tuple(fps))

    # entries carried over unchanged come back as copies, as from the xla path's scan:
    # never the caller's own arrays, which a caller may have donated
    same = jnp.copy
    new_carries, recording = [], []
    for l, layer in enumerate(net.layers):
        (state, dk, ck), (ens_state, edk, eck) = layer_carries[l]
        ens_next = {"m_t": final[l]["m"]}
        if specs[l].high_pass:
            ens_next["r_t"] = final[l]["r"]
        layer_state = ({"a_t": final[l]["a"]}, same(dk), same(ck))
        new_carries.append((layer_state, (ens_next, same(edk), same(eck))))
        ens_rec = {k: recs[l][k] for k in RECORDINGS if k in ensemble_monitor}
        recording.append(({"activity": recs[l]["activity"]}, ens_rec))
    return (new_carries, same(key), same(const_key)), recording
