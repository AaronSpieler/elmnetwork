"""The fused ELM kernels, in Triton (launched by triton_launch.launch).

Seven kernels: the neuron forward and backward, the list forward and backward of pooled wiring,
the input gradient, and the gradient window's list and mlp weight gradients.

Layout (decided on the host, fused_sequence.py): the batch is innermost ("lanes", padded to a
power of two). A buffer is (rows, lanes), or with a time axis (step, rows, lanes): `step` moves
a pointer to a time step in 64 bits, `rows` gives the int32 offsets within a step.

How a kernel learns its variant, all at compile time (each variant is its own specialized
kernel, without branches):
  - shapes and settings come as one frozen dataclass (Neurons, Lists, Lanes), a tl.constexpr;
  - a buffer a variant does not use is passed as None, and `if ptr is not None` is the mode
    switch (a recording, dropout, high-pass filter, pooled or random wiring);
  - buffers that belong together travel as a tuple (the weights, their gradient sums, the
    gradient window's kept inputs), indexed by name; which of them a variant touches follows
    from the layer's description, so the unused members are dummies, never read or written;
  - a store rounds to bf16 when its buffer is bf16 (store_as), so storage dtypes need no flags.
"""

import dataclasses
import os

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

########## COMPILE-TIME DESCRIPTIONS ##########


@dataclasses.dataclass(frozen=True)
class Lanes:
    """The batch as a kernel sees it."""

    lanes: int  # lanes of one program (a power of two)
    stride: int  # lanes of the whole padded batch: the row length in memory
    num_batch: int  # real samples (lanes beyond them are padding)
    grouped: bool  # several sample groups: grid axis 1 picks this program's lanes


@dataclasses.dataclass(frozen=True)
class Neurons:
    """A layer's neurons: sizes (real, and padded to the kernel tiles), math and pieces."""

    num_input: int
    num_neuron: int
    num_branch: int
    nbp: int  # branches, padded
    num_syn: int  # synapses per branch
    num_memory: int
    mp: int  # memory units, padded
    num_hidden: int  # mlp hidden units (depth 1)
    hp: int  # hidden units, padded
    depth: int  # mlp hidden layers: 0 or 1
    mlp_act: int  # MLP_ACT_RELU, MLP_ACT_RELU_SQ or MLP_ACT_SILU
    activation: int  # ACT_RELU, ACT_SPIKE or ACT_IDENTITY
    threshold: float  # spikes (fast_sigmoid)
    scale: float  # spikes: the surrogate gradient's scale
    kappa_r: float  # high-pass EMA decay (when rhist_ptr is not None)
    gain: float  # input_scale of the branch sums
    hc: int  # mlp rows per piece
    pieces: int  # mlp pieces (depth 0: 0 = the whole mlp at once)
    unroll: int  # pieces unrolled by the compiler
    syn_chunk: int  # random wiring: synapse slots gathered at once
    syn_chunks: int
    window: int  # gradient window in steps (1: none)


@dataclasses.dataclass(frozen=True)
class Lists:
    """Pooled wiring: shared lists (a branch, a group of neurons) of synapse sources."""

    num_input: int
    num_neuron: int
    num_branch: int
    nbp: int
    num_syn: int  # slots of a list
    pp: int  # slots, padded
    gp: int  # members (neurons) of a list, padded
    num_runs: int  # lists per branch
    gain: float
    window: int


ACT_RELU, ACT_SPIKE, ACT_IDENTITY = 0, 1, 2
MLP_ACT_RELU, MLP_ACT_RELU_SQ, MLP_ACT_SILU = 0, 1, 2
CTANH_SCALE = tl.constexpr(1.7159)  # custom_tanh: tanh(2x/3) * 1.7159

########## STAND-INS FOR TRITON'S CPU INTERPRETER (THE TESTS) ##########

# The interpreter has no libdevice, truncates float32 -> bf16 and would multiply bf16 operands'
# raw bits: the three functions below are the only places that know about it.
INTERPRETED = tl.constexpr(os.environ.get("TRITON_INTERPRET", "0") not in ("", "0"))


@triton.jit
def to_bf16_rne(x):
    """float32 -> bfloat16, rounding to nearest even (as XLA): the GPU's own conversion
    (cvt.rn); in the interpreter spelled out with integer bit ops."""
    if INTERPRETED:
        bits = x.to(tl.int32, bitcast=True)
        rounded = bits + 0x7FFF + ((bits >> 16) & 1)
        return (rounded >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    else:
        return x.to(tl.bfloat16)


@triton.jit
def tanh(x):
    """tanh: libdevice's on the GPU (as XLA uses), in the interpreter from exp."""
    if INTERPRETED:
        e = tl.exp(-2.0 * tl.abs(x))
        t = (1.0 - e) / (1.0 + e)
        return tl.where(x < 0.0, -t, t)
    else:
        return libdevice.tanh(x)


@triton.jit
def dot(a, b, MODE: tl.constexpr):
    """(m, k) . (k, n) in float32, at the precision MODE ("ieee": exact float32, "tf32",
    "tf32x3", "bf16": operands rounded to bf16, float32 sums), on the tensor cores; tiles with a
    dim below 8 as an exact product sum."""
    if (a.shape[0] < 8) or (a.shape[1] < 8) or (b.shape[1] < 8):
        return product_sum(a, b)
    elif MODE != "bf16":
        return tl.dot(a, b, input_precision=MODE, out_dtype=tl.float32)
    elif INTERPRETED:
        # the bf16-rounded operands' exact products, summed in float32: what the tensor cores
        # compute
        a32 = to_bf16_rne(a).to(tl.float32)
        b32 = to_bf16_rne(b).to(tl.float32)
        if a.shape[1] < 16:
            return product_sum(a32, b32)
        else:
            return tl.dot(a32, b32, input_precision="ieee", out_dtype=tl.float32)
    elif a.shape[1] < 16:
        # bf16 tensor-core dots need k >= 16: k is doubled with interleaved zero columns of a
        # and zero rows of b, which add exact zeros
        a16 = to_bf16_rne(a)
        b16 = to_bf16_rne(b)
        a16 = tl.reshape(tl.join(a16, tl.zeros_like(a16)), (a.shape[0], 2 * a.shape[1]))
        b16 = tl.reshape(tl.permute(tl.join(b16, tl.zeros_like(b16)), (0, 2, 1)),
                         (2 * b.shape[0], b.shape[1]))  # fmt: skip
        return tl.dot(a16, b16, out_dtype=tl.float32)
    else:
        return tl.dot(to_bf16_rne(a), to_bf16_rne(b), out_dtype=tl.float32)


@triton.jit
def product_sum(a, b):
    """(m, k) . (k, n) as an exact float32 broadcast multiply and sum (small tiles)."""
    return tl.sum(a[:, :, None] * b[None, :, :], axis=1)


########## ADDRESSING, LOADS AND STORES ##########


@triton.jit
def program_lanes(B: tl.constexpr):
    """This program's lanes (offsets into a row of B.stride lanes) and which hold samples."""
    lane = tl.arange(0, B.lanes)
    if B.grouped:
        lane = tl.program_id(1) * B.lanes + lane
    return lane, lane < B.num_batch


@triton.jit
def step(ptr, t, WIDTH: tl.constexpr, STRIDE: tl.constexpr):
    """Step t of a (step, WIDTH rows, STRIDE lanes) history; the step offset is 64-bit."""
    return ptr + t.to(tl.int64) * (WIDTH * STRIDE)


@triton.jit
def rows(first, COUNT: tl.constexpr, lane, STRIDE: tl.constexpr):
    """Offsets (COUNT, lanes) of the rows first .. first + COUNT of a (rows, STRIDE lanes)
    buffer, this program's lanes."""
    return (first + tl.arange(0, COUNT))[:, None] * STRIDE + lane[None, :]


@triton.jit
def store_as(ptr, value):
    """Stores float32 values in the buffer's own type: bf16 buffers round to nearest even."""
    if ptr.dtype.element_ty == tl.bfloat16:
        tl.store(ptr, to_bf16_rne(value))
    else:
        tl.store(ptr, value)


@triton.jit
def accumulate(ptr, value, mask):
    """ptr += value (masked): a weight gradient summed over time steps."""
    tl.store(ptr, tl.load(ptr, mask=mask, other=0.0) + value, mask=mask)


@triton.jit
def tile(ROWS: tl.constexpr, COLS: tl.constexpr, n_rows: tl.constexpr, n_cols: tl.constexpr,
         row0=0, col0=0):  # fmt: skip
    """Offsets and mask of a (ROWS, COLS) tile at (row0, col0) of a row-major (n_rows, n_cols)
    matrix: entries past the matrix are masked."""
    r = row0 + tl.arange(0, ROWS)
    c = col0 + tl.arange(0, COLS)
    return r[:, None] * n_cols + c[None, :], (r < n_rows)[:, None] & (c < n_cols)[None, :]


@triton.jit
def load_tile(ptr, ROWS: tl.constexpr, COLS: tl.constexpr, n_rows: tl.constexpr,
              n_cols: tl.constexpr, row0=0, col0=0):  # fmt: skip
    """A tile of a matrix (see tile) as float32, zero past the matrix."""
    offsets, mask = tile(ROWS, COLS, n_rows, n_cols, row0, col0)
    return tl.load(ptr + offsets, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def accumulate_tile(ptr, value, ROWS: tl.constexpr, COLS: tl.constexpr, n_rows: tl.constexpr,
                    n_cols: tl.constexpr, row0=0, col0=0):  # fmt: skip
    """Adds value to a tile of a weight-gradient sum (see tile)."""
    offsets, mask = tile(ROWS, COLS, n_rows, n_cols, row0, col0)
    accumulate(ptr + offsets, value, mask)


@triton.jit
def load_vector(ptr, SIZE: tl.constexpr, n: tl.constexpr):
    i = tl.arange(0, SIZE)
    return tl.load(ptr + i, mask=i < n, other=0.0).to(tl.float32)


@triton.jit
def grad_sum(ptr, size: tl.constexpr, num_neuron: tl.constexpr, B: tl.constexpr):
    """This neuron's block of a weight-gradient sum: (neuron, size), with several sample groups
    (sample group, neuron, size): each group sums its lanes into its own block."""
    if B.grouped:
        ptr += tl.program_id(1) * num_neuron * size
    return ptr + tl.program_id(0) * size


########## INPUT GRADIENT ##########


@triton.jit
def elm_seq_ingrad(
    t_ptr, start_ptr, count_ptr, entry_ptr, grad_ptr, weight_ptr,
    din_ptr, drec_ptr,
    dx_ptr, dacarry_ptr,
    NUM_INPUT: tl.constexpr, NUM_NEURON: tl.constexpr, LANES: tl.constexpr,
    CHUNK: tl.constexpr, DX_HISTORY: tl.constexpr,
):  # fmt: skip
    """One program per input slot j (layer inputs, then the layer's own neurons): the cotangent
    of that slot at step t, summed over the synapse entries that read it (an inverse wiring:
    entries start..start+count). Random wiring weights each entry's branch cotangent by its
    synapse weight; pooled wiring (weight_ptr None) has entries that already hold weighted
    input gradients. Layer inputs (j < NUM_INPUT) write dx (DX_HISTORY: step t of the network
    input's gradient history); recurrent slots write the cotangent of a_{t-1}. din / drec: the
    dropout masks, or None."""
    t = tl.load(t_ptr)
    j = tl.program_id(0)
    start = tl.load(start_ptr + j)
    count = tl.load(count_ptr + j)
    lane = tl.arange(0, LANES)
    offsets = tl.arange(0, CHUNK)
    acc = tl.zeros((LANES,), tl.float32)
    for c in range(0, tl.cdiv(count, CHUNK)):
        valid = c * CHUNK + offsets < count
        pos = start + c * CHUNK + offsets
        entry = tl.load(entry_ptr + pos, mask=valid, other=0)
        grad = tl.load(grad_ptr + entry[:, None] * LANES + lane[None, :], mask=valid[:, None],
                       other=0.0).to(tl.float32)  # fmt: skip
        if weight_ptr is None:
            acc += tl.sum(grad, axis=0)
        else:
            weight = tl.load(weight_ptr + pos, mask=valid, other=0.0)
            acc += tl.sum(weight[:, None] * grad, axis=0)
    is_input = j < NUM_INPUT
    # masked rows keep in-range indices (row 0), chosen per program, not per lane
    x_index = tl.where(is_input, j, 0) * LANES + lane
    a_index = tl.where(is_input, 0, j - NUM_INPUT) * LANES + lane
    # (masks as lane vectors: the interpreter rejects a vector & a scalar condition)
    slot_j = tl.zeros_like(lane) + j
    x_mask = slot_j < NUM_INPUT
    a_mask = slot_j >= NUM_INPUT
    acc_x = acc
    acc_a = acc
    if din_ptr is not None:  # through the masks of the layer input and of the recurrent input
        acc_x = acc * tl.load(din_ptr + x_index, mask=x_mask, other=0.0)
        acc_a = acc * tl.load(drec_ptr + a_index, mask=a_mask, other=0.0)
    if DX_HISTORY:
        tl.store(step(dx_ptr, t, NUM_INPUT, LANES) + x_index, acc_x, mask=x_mask)
    else:
        tl.store(dx_ptr + x_index, acc_x, mask=x_mask)
    tl.store(dacarry_ptr + a_index, acc_a, mask=a_mask)


########## LIST KERNELS (POOLED WIRING) ##########


@triton.jit
def list_members(src_ptr, neuron_ptr, P: tl.constexpr):
    """This program's shared list (branch k, group of neurons): its sources and which slots are
    real, its members' rows in the synapse weights (missing members: a row past the end) and
    in the per-neuron branch buffers (missing members: the dump row past the end)."""
    lst = tl.program_id(0)
    k = lst // P.num_runs
    j = tl.arange(0, P.pp)
    real = j < P.num_syn
    src = tl.load(src_ptr + lst * P.num_syn + j, mask=real, other=0)
    neuron = tl.load(neuron_ptr + lst * P.gp + tl.arange(0, P.gp))
    member = neuron >= 0
    # rows are selected before the multiply by the row length, so Triton sees contiguous,
    # aligned rows and emits vector loads and stores
    weight_row = tl.where(member, neuron * P.num_branch + k, P.num_neuron * P.num_branch)
    branch_row = tl.where(member, neuron * P.nbp + k, P.num_neuron * P.nbp)
    return src, real, weight_row, member, branch_row


@triton.jit
def list_weights(ws_ptr, weight_row, member, real, P: tl.constexpr):
    """The members' synapse weights for the list's slots (member, slot), zero where absent."""
    ptrs = ws_ptr + (weight_row * P.num_syn)[:, None] + tl.arange(0, P.pp)[None, :]
    return tl.load(ptrs, mask=member[:, None] & real[None, :], other=0.0).to(tl.float32)


@triton.jit
def list_inputs(uhist_ptr, t, src, real, lane, in_batch, P: tl.constexpr, B: tl.constexpr):
    """The list's inputs (slot, lanes) at step t, from the layer's input history."""
    row = tl.where(real, src, 0)
    ptrs = (step(uhist_ptr, t, P.num_input + P.num_neuron, B.stride)
            + row[:, None] * B.stride + lane[None, :])  # fmt: skip
    return tl.load(ptrs, mask=real[:, None] & in_batch[None, :], other=0.0).to(tl.float32)


@triton.jit
def accumulate_weights(acc_ptr, weight_row, member, real, dw, P: tl.constexpr,
                       B: tl.constexpr):  # fmt: skip
    """Adds dw (member, slot) to this sample group's row of the synapse-weight sums (real
    members and slots only); a row is every synapse plus a dump slot."""
    acc = acc_ptr + (weight_row * P.num_syn)[:, None] + tl.arange(0, P.pp)[None, :]
    if B.grouped:
        acc += tl.program_id(1) * (P.num_neuron * P.num_branch * P.num_syn + P.pp)
    accumulate(acc, dw, member[:, None] & real[None, :])


@triton.jit
def elm_seq_pool_fwd(
    t_ptr, uhist_ptr, src_ptr, neuron_ptr, ws_ptr,
    out_ptr,
    P: tl.constexpr, B: tl.constexpr, DOT: tl.constexpr, HISTORY: tl.constexpr,
):  # fmt: skip
    """Branch sums of one shared list's members at step t: the list's inputs gathered once, one
    (member x slot) . (slot x lanes) product, times input_scale. Written to the branch-sum
    buffer, or with HISTORY (the gradient window) into step t of the branch-sum history."""
    t = tl.load(t_ptr)
    lane, in_batch = program_lanes(B)
    src, real, weight_row, member, branch_row = list_members(src_ptr, neuron_ptr, P)
    x = list_inputs(uhist_ptr, t, src, real, lane, in_batch, P, B)
    w = list_weights(ws_ptr, weight_row, member, real, P)
    b = dot(w, x, DOT) * P.gain
    out = out_ptr + branch_row[:, None] * B.stride + lane[None, :]
    if HISTORY:
        out += t.to(tl.int64) * ((P.num_neuron * P.nbp + 1) * B.stride)
    store_as(out, b)


@triton.jit
def elm_seq_pool_bwd(
    t_ptr, uhist_ptr, src_ptr, neuron_ptr, ws_ptr, grad_ptr,
    acc_ptr,
    dlist_ptr,
    P: tl.constexpr, B: tl.constexpr, DOT: tl.constexpr,
):  # fmt: skip
    """One shared list at step t: the input gradients of its entries (list, slot), which the
    input-gradient kernel sums per input, and its members' synapse-weight gradients, added to
    this sample group's sums. The branch cotangents come from the neuron backward's buffer; with
    a gradient window (P.window > 1) from slot t % window of their history, and the weight
    gradients are then elm_seq_pool_wgrad's (acc_ptr and uhist_ptr are None)."""
    t = tl.load(t_ptr)
    lane, in_batch = program_lanes(B)
    src, real, weight_row, member, branch_row = list_members(src_ptr, neuron_ptr, P)
    w = list_weights(ws_ptr, weight_row, member, real, P)
    grad = grad_ptr + branch_row[:, None] * B.stride + lane[None, :]
    if P.window > 1:
        grad += (t % P.window).to(tl.int64) * (P.num_neuron * P.nbp * B.stride)
    db = tl.load(grad, mask=member[:, None] & in_batch[None, :], other=0.0).to(tl.float32)
    if acc_ptr is not None:
        x = list_inputs(uhist_ptr, t, src, real, lane, in_batch, P, B)
        accumulate_weights(acc_ptr, weight_row, member, real, dot(db, tl.trans(x), DOT), P, B)
    dx = dot(tl.trans(w), db, DOT)
    entry = tl.program_id(0) * P.pp + tl.arange(0, P.pp)
    tl.store(dlist_ptr + entry[:, None] * B.stride + lane[None, :], dx)


@triton.jit
def elm_seq_pool_wgrad(
    t0_ptr, uhist_ptr, src_ptr, neuron_ptr, grad_ptr,
    acc_ptr,
    P: tl.constexpr, B: tl.constexpr, DOT: tl.constexpr,
):  # fmt: skip
    """Gradient window: one shared list's synapse-weight gradients over the window of steps
    t0 .. t0 + window - 1 (slot s of the kept branch cotangents is step t0 + s), summed in the
    backward's step order (latest first) and added to this sample group's sums."""
    t0 = tl.load(t0_ptr)
    lane, in_batch = program_lanes(B)
    src, real, weight_row, member, branch_row = list_members(src_ptr, neuron_ptr, P)
    dmask = member[:, None] & in_batch[None, :]
    grad = grad_ptr + branch_row[:, None] * B.stride + lane[None, :]
    dw = tl.zeros((P.gp, P.pp), tl.float32)
    for i in range(0, P.window):
        s = P.window - 1 - i
        x = list_inputs(uhist_ptr, t0 + s, src, real, lane, in_batch, P, B)
        db = tl.load(step(grad, s, P.num_neuron * P.nbp, B.stride), mask=dmask,
                     other=0.0).to(tl.float32)  # fmt: skip
        dw += dot(db, tl.trans(x), DOT)
    accumulate_weights(acc_ptr, weight_row, member, real, dw, P, B)


########## NEURON MATH, FORWARD ##########

# The neuron kernels' weights travel as one tuple, indexed by:
#   CONN, WS (random wiring: sources and weights of the synapses), W1B, W1M, B1, W2 (mlp depth
#   1), W2B, W2M (depth 0), B2, WY, KM, KL, BIAS
CONN, WS, W1B, W1M, B1, W2, W2B, W2M, B2, WY, KM, KL, BIAS = map(tl.constexpr, range(13))
# and their gradient sums as another (there is none for CONN), indexed by:
G_WS, G_W1B, G_W1M, G_B1, G_W2, G_W2B, G_W2M, G_B2, G_WY, G_KM, G_KL, G_BIAS = map(
    tl.constexpr, range(12))  # fmt: skip
# the gradient window's kept inputs: the branch cotangents, dmo, and the hidden layer's h, dhpre
K_DB, K_DMO, K_H, K_DH = map(tl.constexpr, range(4))


@triton.jit
def gathered_branch_sums(uhist_ptr, weights, n, t, lane, in_batch, L: tl.constexpr,
                         B: tl.constexpr):  # fmt: skip
    """Random wiring: sum over synapses of weight x input at step t, per branch (branch, lanes),
    gathered from the input history in pieces of L.syn_chunk slots (only pieces holding real
    synapses)."""
    b = tl.arange(0, L.nbp)
    acc = tl.zeros((L.nbp, B.lanes), tl.float32)
    base = n * L.num_branch * L.num_syn
    for c in range(0, L.syn_chunks):
        p = c * L.syn_chunk + tl.arange(0, L.syn_chunk)
        valid = (b < L.num_branch)[:, None] & (p < L.num_syn)[None, :]
        idx = base + b[:, None] * L.num_syn + p[None, :]
        conn = tl.load(weights[CONN] + idx, mask=valid, other=0)
        ws = tl.load(weights[WS] + idx, mask=valid, other=0.0).to(tl.float32)
        row = tl.where(valid, conn, 0)  # per row, so the lanes stay contiguous (vector loads)
        u_ptrs = (step(uhist_ptr, t, L.num_input + L.num_neuron, B.stride)
                  + row[:, :, None] * B.stride + lane[None, None, :])  # fmt: skip
        u = tl.load(u_ptrs, mask=valid[:, :, None] & in_batch[None, None, :], other=0.0)
        acc += tl.sum(ws[:, :, None] * u.to(tl.float32), axis=1)
    return acc


@triton.jit
def branch_sums(uhist_ptr, bsum_ptr, weights, n, t, lane, in_batch, L: tl.constexpr,
                B: tl.constexpr):  # fmt: skip
    """The neuron's branch sums (branch, lanes) at step t: gathered (random wiring, bsum_ptr
    None), or the list kernel's, from its buffer or with a gradient window from step t of its
    history (a step holds every neuron's branches and the lists' dump row)."""
    if bsum_ptr is None:
        bact = gathered_branch_sums(uhist_ptr, weights, n, t, lane, in_batch, L, B) * L.gain
    else:
        ptrs = bsum_ptr + rows(n * L.nbp, L.nbp, lane, B.stride)
        if L.window > 1:
            ptrs = step(ptrs, t, L.num_neuron * L.nbp + 1, B.stride)
        mask = (tl.arange(0, L.nbp) < L.num_branch)[:, None] & in_batch[None, :]
        bact = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    return bact


@triton.jit
def previous_state(mhist_ptr, rhist_ptr, n, t, lane, in_batch, L: tl.constexpr,
                   B: tl.constexpr):  # fmt: skip
    """The state before step t: the memory (memory, lanes) and, with high-pass filter
    (rhist_ptr not None), the running EMA r_t (lanes)."""
    m = tl.arange(0, L.mp)
    mp = tl.load(step(mhist_ptr, t, L.num_neuron * L.mp, B.stride)
                 + rows(n * L.mp, L.mp, lane, B.stride),
                 mask=(m < L.num_memory)[:, None] & in_batch[None, :], other=0.0)  # fmt: skip
    vec = n * B.stride + lane
    r_p = tl.zeros_like(lane).to(tl.float32)
    if rhist_ptr is not None:
        r_p = tl.load(step(rhist_ptr, t, L.num_neuron, B.stride) + vec, mask=in_batch,
                      other=0.0)  # fmt: skip
    return mp, r_p


@triton.jit
def neuron_params(weights, n, L: tl.constexpr):
    """The neuron's per-unit vectors and bias: km, kl, wy, bias."""
    km = load_vector(weights[KM] + n * L.num_memory, L.mp, L.num_memory)
    kl = load_vector(weights[KL] + n * L.num_memory, L.mp, L.num_memory)
    wy = load_vector(weights[WY] + n * L.num_memory, L.mp, L.num_memory)
    bias = tl.load(weights[BIAS] + n)
    return km, kl, wy, bias


@triton.jit
def hidden_weights(weights, n, h0, L: tl.constexpr):
    """A neuron's first-layer weights of hidden rows h0 .. h0 + L.hc (zero past the real
    rows), and their second-layer columns."""
    w1b = load_tile(weights[W1B] + n * L.num_hidden * L.num_branch, L.hc, L.nbp, L.num_hidden,
                    L.num_branch, h0)  # fmt: skip
    w1m = load_tile(weights[W1M] + n * L.num_hidden * L.num_memory, L.hc, L.mp, L.num_hidden,
                    L.num_memory, h0)  # fmt: skip
    i = h0 + tl.arange(0, L.hc)
    b1 = tl.load(weights[B1] + n * L.num_hidden + i, mask=i < L.num_hidden, other=0.0)
    w2 = load_tile(weights[W2] + n * L.num_memory * L.num_hidden, L.mp, L.hc, L.num_memory,
                   L.num_hidden, 0, h0)  # fmt: skip
    return w1b, w1m, b1, w2


@triton.jit
def hidden_piece(w1b, w1m, b1, bact, md, L: tl.constexpr, DOT: tl.constexpr):
    """Hidden rows of the mlp: pre-activation (or relu(z) for relu_squared) and activation h."""
    z = dot(w1b, bact, DOT) + dot(w1m, md, DOT) + b1[:, None]
    if L.mlp_act == 0:  # relu
        return z, tl.maximum(z, 0.0)
    elif L.mlp_act == 1:  # relu_squared
        r = tl.maximum(z, 0.0)
        return r, r * r
    else:  # silu
        return z, z * tl.sigmoid(z)


@triton.jit
def memory_weights(weights, n, r0, RC: tl.constexpr, L: tl.constexpr):
    """Depth 0: rows r0 .. r0 + RC of the mlp's weights w2 [branches; memory]."""
    w2b = load_tile(weights[W2B] + n * L.num_memory * L.num_branch, RC, L.nbp, L.num_memory,
                    L.num_branch, r0)  # fmt: skip
    w2m = load_tile(weights[W2M] + n * L.num_memory * L.num_memory, RC, L.mp, L.num_memory,
                    L.num_memory, r0)  # fmt: skip
    return w2b, w2m


@triton.jit
def placement(r0, RC: tl.constexpr, MP: tl.constexpr):
    """Depth 0 in pieces: the 0/1 matrix (memory, piece row) that places the rows r0 .. r0 + RC
    into a memory tile (its transpose picks them out), by an exact product."""
    return (tl.arange(0, MP)[:, None] == r0 + tl.arange(0, RC)[None, :]).to(tl.float32)


@triton.jit
def mlp_forward(weights, bact, md, n, L: tl.constexpr, DOT: tl.constexpr):
    """The mlp's output mo (memory, lanes) from the branch sums and the decayed memory."""
    b2 = load_vector(weights[B2] + n * L.num_memory, L.mp, L.num_memory)
    mo = tl.zeros_like(md) + b2[:, None]
    if L.depth == 1:
        for c in tl.range(0, L.pieces, loop_unroll_factor=L.unroll):
            w1b, w1m, b1, w2 = hidden_weights(weights, n, c * L.hc, L)
            _, h = hidden_piece(w1b, w1m, b1, bact, md, L, DOT)
            mo += dot(w2, h, DOT)
    elif L.pieces == 0:
        w2b, w2m = memory_weights(weights, n, 0, L.mp, L)
        mo = dot(w2b, bact, DOT) + dot(w2m, md, DOT) + b2[:, None]
    else:
        for c in tl.range(0, L.pieces, loop_unroll_factor=L.unroll):
            w2b, w2m = memory_weights(weights, n, c * L.hc, L.hc, L)
            part = dot(w2b, bact, DOT) + dot(w2m, md, DOT)
            mo += dot(placement(c * L.hc, L.hc, L.mp), part, "ieee")
    return mo


@triton.jit
def neuron_output(mo, md, kl, wy, r_p, bias, HIGH_PASS: tl.constexpr, L: tl.constexpr):
    """From the mlp output to the new memory, readout, high-pass preact and activity:
    returns th, mn, readout, r_n, preact, z, a."""
    th = tanh(mo * 2.0 / 3.0)
    dm = th * CTANH_SCALE
    mn = md + (1.0 - kl[:, None]) * dm
    readout = tl.sum(wy[:, None] * mn, axis=0)
    if HIGH_PASS:
        r_n = L.kappa_r * r_p + (1.0 - L.kappa_r) * readout
        preact = readout - r_n
    else:
        r_n = readout
        preact = readout
    z = preact + bias
    if L.activation == 0:  # relu
        a = tl.maximum(z, 0.0)
    elif L.activation == 1:  # spike: fast_sigmoid's forward step
        a = (z - L.threshold > 0).to(tl.float32)
    else:  # identity
        a = z
    return th, mn, readout, r_n, preact, z, a


@triton.jit
def elm_seq_fwd(
    # read only
    t_ptr, drec_ptr, dnext_ptr, bsum_ptr, weights,
    # updated in place: the histories, and the recordings (None: not recorded)
    uhist_ptr, ahist_ptr, mhist_ptr, rhist_ptr, unext_ptr,
    preact_ptr, readout_ptr, mlp_ptr, branch_ptr,
    L: tl.constexpr, B: tl.constexpr, DOT: tl.constexpr,
    NEXT_WIDTH: tl.constexpr,
):  # fmt: skip
    """One neuron (program axis 0) and sample group (axis 1) at step t: branch sums, the mlp,
    the memory update, the readout, optional high-pass filter and the activity. Writes the new
    states into step t + 1 of their histories, the layer's recurrent input into step t + 1 of
    its input history, the next layer's input into step t of that one's history, and the
    recordings into step t."""
    t = tl.load(t_ptr)
    n = tl.program_id(0)
    lane, in_batch = program_lanes(B)
    bact = branch_sums(uhist_ptr, bsum_ptr, weights, n, t, lane, in_batch, L, B)
    mp, r_p = previous_state(mhist_ptr, rhist_ptr, n, t, lane, in_batch, L, B)
    km, kl, wy, bias = neuron_params(weights, n, L)
    md = km[:, None] * mp
    mo = mlp_forward(weights, bact, md, n, L, DOT)
    th, mn, readout, r_n, preact, z, a = neuron_output(
        mo, md, kl, wy, r_p, bias, rhist_ptr is not None, L
    )
    # inputs of the next step (recurrent part) and of the next layer at step t
    vec = n * B.stride + lane
    a_rec = a
    a_next = a
    if drec_ptr is not None:
        a_rec = a * tl.load(drec_ptr + vec)
    if dnext_ptr is not None:
        a_next = a * tl.load(dnext_ptr + vec)
    store_as(step(uhist_ptr, t + 1, L.num_input + L.num_neuron, B.stride)
             + (L.num_input + n) * B.stride + lane, a_rec)  # fmt: skip
    if unext_ptr is not None:
        store_as(step(unext_ptr, t, NEXT_WIDTH, B.stride) + vec, a_next)
    # states into step t + 1, recordings into step t
    m_rows = rows(n * L.mp, L.mp, lane, B.stride)
    tl.store(step(ahist_ptr, t + 1, L.num_neuron, B.stride) + vec, a)
    tl.store(step(mhist_ptr, t + 1, L.num_neuron * L.mp, B.stride) + m_rows, mn)
    if rhist_ptr is not None:
        tl.store(step(rhist_ptr, t + 1, L.num_neuron, B.stride) + vec, r_n)
    if preact_ptr is not None:
        tl.store(step(preact_ptr, t, L.num_neuron, B.stride) + vec, preact)
    if readout_ptr is not None:
        tl.store(step(readout_ptr, t, L.num_neuron, B.stride) + vec, readout)
    if mlp_ptr is not None:
        tl.store(step(mlp_ptr, t, L.num_neuron * L.mp, B.stride) + m_rows, mo)
    if branch_ptr is not None:
        tl.store(step(branch_ptr, t, L.num_neuron * L.nbp, B.stride)
                 + rows(n * L.nbp, L.nbp, lane, B.stride), bact)  # fmt: skip


########## NEURON MATH, BACKWARD ##########


@triton.jit
def output_backward(ca, ct_preact, ct_r, ct_readout, ct_m, z, mn, th, wy, kl,
                    HIGH_PASS: tl.constexpr, L: tl.constexpr):  # fmt: skip
    """From the activity's cotangent back to the mlp output's: returns dmo (without the mlp
    recording's share), dmn, d_r_prev and the gradients of bias, wy and kl."""
    if L.activation == 0:  # relu
        dz = tl.where(z > 0.0, ca, 0.0)
    elif L.activation == 1:  # fast_sigmoid's surrogate gradient
        s = L.scale * tl.abs(z - L.threshold) + 1.0
        dz = ca / (s * s)
    else:  # identity
        dz = ca
    g_bias = tl.sum(dz, axis=0)
    dpreact = dz + ct_preact
    if HIGH_PASS:
        dr = ct_r - dpreact
        dreadout = dpreact + (1.0 - L.kappa_r) * dr
        d_r_prev = L.kappa_r * dr
    else:
        dreadout = dpreact
        d_r_prev = dpreact * 0.0
    dreadout = dreadout + ct_readout
    g_wy = tl.sum(dreadout[None, :] * mn, axis=1)
    dmn = ct_m + wy[:, None] * dreadout[None, :]
    g_kl = -tl.sum(dmn * (th * CTANH_SCALE), axis=1)
    dth = (dmn * (1.0 - kl[:, None])) * CTANH_SCALE
    dmo = (dth + dth * th) * (1.0 - th) / 3.0 * 2.0
    return dmo, dmn, d_r_prev, g_bias, g_wy, g_kl


@triton.jit
def keep_rows(ptr, value, t, first, WIDTH: tl.constexpr, lane, L: tl.constexpr,
              B: tl.constexpr):  # fmt: skip
    """Gradient window: value's rows, starting at row `first` of a step's WIDTH rows, into slot
    t % window of a kept-inputs history, for the window's deferred weight gradients."""
    store_as(step(ptr, t % L.window, WIDTH, B.stride)
             + rows(first, value.shape[0], lane, B.stride), value)  # fmt: skip


@triton.jit
def kept_rows(ptr, s, first, COUNT: tl.constexpr, WIDTH: tl.constexpr, lane, B: tl.constexpr):
    """COUNT rows from row `first`, of slot s of a kept-inputs history (see keep_rows)."""
    return tl.load(step(ptr, s, WIDTH, B.stride) + rows(first, COUNT, lane, B.stride)).to(
        tl.float32)  # fmt: skip


@triton.jit
def hidden_backward_piece(weights, acc, kept, bact, md, dmo, n, t, h0, lane, L: tl.constexpr,
                          B: tl.constexpr, DOT: tl.constexpr):  # fmt: skip
    """Backward through hidden rows h0 .. h0 + L.hc: their bias gradient, the weight gradients
    of w1 and w2 (with a gradient window: their inputs h and dhpre kept instead), and their
    share of the branch and memory cotangents (returned)."""
    w1b, w1m, b1, w2 = hidden_weights(weights, n, h0, L)
    aux, h = hidden_piece(w1b, w1m, b1, bact, md, L, DOT)
    if L.window > 1:
        keep_rows(kept[K_H], h, t, n * L.hp + h0, L.num_neuron * L.hp, lane, L, B)
    else:
        accumulate_tile(grad_sum(acc[G_W2], L.num_memory * L.num_hidden, L.num_neuron, B),
                        dot(dmo, tl.trans(h), DOT), L.mp, L.hc, L.num_memory, L.num_hidden,
                        0, h0)  # fmt: skip
    dh = dot(tl.trans(w2), dmo, DOT)
    if L.mlp_act == 0:  # relu
        dhpre = tl.where(aux > 0.0, dh, 0.0)
    elif L.mlp_act == 1:  # relu_squared: aux is relu(z)
        dhpre = dh * 2.0 * aux
    else:  # silu: aux is z, d/dz(z * sig(z)) = sig(z) * (1 + z * (1 - sig(z)))
        s = tl.sigmoid(aux)
        dhpre = dh * s * (1.0 + aux * (1.0 - s))
    i = h0 + tl.arange(0, L.hc)
    accumulate(grad_sum(acc[G_B1], L.num_hidden, L.num_neuron, B) + i, tl.sum(dhpre, axis=1),
               i < L.num_hidden)  # fmt: skip
    if L.window > 1:
        keep_rows(kept[K_DH], dhpre, t, n * L.hp + h0, L.num_neuron * L.hp, lane, L, B)
    else:
        accumulate_tile(grad_sum(acc[G_W1B], L.num_hidden * L.num_branch, L.num_neuron, B),
                        dot(dhpre, tl.trans(bact), DOT), L.hc, L.nbp, L.num_hidden,
                        L.num_branch, h0)  # fmt: skip
        accumulate_tile(grad_sum(acc[G_W1M], L.num_hidden * L.num_memory, L.num_neuron, B),
                        dot(dhpre, tl.trans(md), DOT), L.hc, L.mp, L.num_hidden,
                        L.num_memory, h0)  # fmt: skip
    return dot(tl.trans(w1b), dhpre, DOT), dot(tl.trans(w1m), dhpre, DOT)


@triton.jit
def memory_backward_piece(weights, acc, bact, md, dmo, n, r0, RC: tl.constexpr,
                          L: tl.constexpr, B: tl.constexpr, DOT: tl.constexpr):  # fmt: skip
    """Depth 0: backward through the memory rows r0 .. r0 + RC of w2 [branches; memory]."""
    w2b, w2m = memory_weights(weights, n, r0, RC, L)
    if L.window == 1:
        accumulate_tile(grad_sum(acc[G_W2B], L.num_memory * L.num_branch, L.num_neuron, B),
                        dot(dmo, tl.trans(bact), DOT), RC, L.nbp, L.num_memory, L.num_branch,
                        r0)  # fmt: skip
        accumulate_tile(grad_sum(acc[G_W2M], L.num_memory * L.num_memory, L.num_neuron, B),
                        dot(dmo, tl.trans(md), DOT), RC, L.mp, L.num_memory, L.num_memory,
                        r0)  # fmt: skip
    return dot(tl.trans(w2b), dmo, DOT), dot(tl.trans(w2m), dmo, DOT)


@triton.jit
def mlp_backward(weights, acc, kept, bact, md, dmo, n, t, lane, L: tl.constexpr,
                 B: tl.constexpr, DOT: tl.constexpr):  # fmt: skip
    """Backward through the mlp, in the forward's pieces."""
    dbact = tl.zeros_like(bact)
    dmd_mlp = tl.zeros_like(md)
    if L.depth == 1:
        for c in tl.range(0, L.pieces, loop_unroll_factor=L.unroll):
            db_c, dm_c = hidden_backward_piece(weights, acc, kept, bact, md, dmo, n, t,
                                               c * L.hc, lane, L, B, DOT)  # fmt: skip
            dbact += db_c
            dmd_mlp += dm_c
    elif L.pieces == 0:
        dbact, dmd_mlp = memory_backward_piece(weights, acc, bact, md, dmo, n, 0, L.mp, L, B, DOT)
    else:
        for c in tl.range(0, L.pieces, loop_unroll_factor=L.unroll):
            dmo_c = dot(tl.trans(placement(c * L.hc, L.hc, L.mp)), dmo, "ieee")
            db_c, dm_c = memory_backward_piece(weights, acc, bact, md, dmo_c, n, c * L.hc,
                                               L.hc, L, B, DOT)  # fmt: skip
            dbact += db_c
            dmd_mlp += dm_c
    return dbact, dmd_mlp


@triton.jit
def synapse_weight_grads(acc_ptr, uhist_ptr, weights, dbsum, n, t, lane, in_batch,
                         L: tl.constexpr, B: tl.constexpr):  # fmt: skip
    """Random wiring: each synapse's weight gradient, the lane sum of its branch cotangent times
    its input (re-gathered, in pieces of L.syn_chunk slots), added to the sums."""
    b = tl.arange(0, L.nbp)
    acc = grad_sum(acc_ptr, L.num_branch * L.num_syn, L.num_neuron, B)
    for c in range(0, L.syn_chunks):
        p = c * L.syn_chunk + tl.arange(0, L.syn_chunk)
        valid = (b < L.num_branch)[:, None] & (p < L.num_syn)[None, :]
        idx = b[:, None] * L.num_syn + p[None, :]
        conn = tl.load(weights[CONN] + n * L.num_branch * L.num_syn + idx, mask=valid, other=0)
        row = tl.where(valid, conn, 0)
        u_ptrs = (step(uhist_ptr, t, L.num_input + L.num_neuron, B.stride)
                  + row[:, :, None] * B.stride + lane[None, None, :])  # fmt: skip
        u = tl.load(u_ptrs, mask=valid[:, :, None] & in_batch[None, None, :], other=0.0)
        accumulate(acc + idx, tl.sum(dbsum[:, None, :] * u.to(tl.float32), axis=2), valid)


@triton.jit
def elm_seq_bwd(
    # read only: the step's inputs, the cotangents of the activity and of the recordings (None:
    # not recorded)
    t_ptr, uhist_ptr, mhist_ptr, rhist_ptr, bsum_ptr, cact_ptr, dacarry_ptr,
    dxbuf_ptr, cpreact, creadout, cmlp, cbranch, cmemory,
    weights,
    # updated in place: the state cotangents, the gradient window's kept inputs (dbhist,
    # dmohist, hhist, dhhist), the weight-gradient sums
    dmcarry_ptr, drcarry_ptr, kept, acc,
    # written: the branch cotangents (without a gradient window)
    dbs_ptr,
    L: tl.constexpr, B: tl.constexpr, DOT: tl.constexpr,
):  # fmt: skip
    """One neuron and sample group at step t, backward."""
    t = tl.load(t_ptr)
    n = tl.program_id(0)
    lane, in_batch = program_lanes(B)
    # the forward, as elm_seq_fwd
    bact = branch_sums(uhist_ptr, bsum_ptr, weights, n, t, lane, in_batch, L, B)
    mp, r_p = previous_state(mhist_ptr, rhist_ptr, n, t, lane, in_batch, L, B)
    km, kl, wy, bias = neuron_params(weights, n, L)
    md = km[:, None] * mp
    mo = mlp_forward(weights, bact, md, n, L, DOT)
    th, mn, readout, r_n, preact, z, a = neuron_output(
        mo, md, kl, wy, r_p, bias, rhist_ptr is not None, L
    )
    # cotangents of the new state and of the recordings (zero on padded lanes)
    vec = n * B.stride + lane
    b_rows = rows(n * L.nbp, L.nbp, lane, B.stride)
    m_rows = rows(n * L.mp, L.mp, lane, B.stride)
    zero = tl.zeros_like(lane).to(tl.float32)
    ca = tl.load(step(cact_ptr, t, L.num_neuron, B.stride) + vec) + tl.load(dacarry_ptr + vec)
    if dxbuf_ptr is not None:
        ca += tl.load(dxbuf_ptr + vec)
    ct_m = tl.load(dmcarry_ptr + m_rows)
    if cmemory is not None:
        ct_m += tl.load(step(cmemory, t, L.num_neuron * L.mp, B.stride) + m_rows)
    ct_r = zero
    if rhist_ptr is not None:
        ct_r = tl.load(drcarry_ptr + vec)
    ct_preact = zero
    if cpreact is not None:
        ct_preact = tl.load(step(cpreact, t, L.num_neuron, B.stride) + vec)
    ct_readout = zero
    if creadout is not None:
        ct_readout = tl.load(step(creadout, t, L.num_neuron, B.stride) + vec)
    # back to the mlp output
    dmo, dmn, d_r_prev, g_bias, g_wy, g_kl = output_backward(
        ca, ct_preact, ct_r, ct_readout, ct_m, z, mn, th, wy, kl,
        rhist_ptr is not None, L,
    )
    if cmlp is not None:
        dmo += tl.load(step(cmlp, t, L.num_neuron * L.mp, B.stride) + m_rows)
    if L.window > 1:
        keep_rows(kept[K_DMO], dmo, t, n * L.mp, L.num_neuron * L.mp, lane, L, B)
    # through the mlp, to the branch sums and the previous memory
    dbact, dmd_mlp = mlp_backward(weights, acc, kept, bact, md, dmo, n, t, lane, L, B, DOT)
    if cbranch is not None:
        dbact += tl.load(step(cbranch, t, L.num_neuron * L.nbp, B.stride) + b_rows)
    dmd = dmn + dmd_mlp
    tl.store(dmcarry_ptr + m_rows, dmd * km[:, None])
    if rhist_ptr is not None:
        tl.store(drcarry_ptr + vec, d_r_prev)
    dbsum = dbact * L.gain  # the synaptic (branch) cotangent
    if L.window > 1:
        keep_rows(kept[K_DB], dbsum, t, n * L.nbp, L.num_neuron * L.nbp, lane, L, B)
    else:
        tl.store(dbs_ptr + b_rows, dbsum)
    # the remaining weight gradients, added to their sums
    if bsum_ptr is None:  # random wiring (pooled: the list kernels')
        synapse_weight_grads(acc[G_WS], uhist_ptr, weights, dbsum, n, t, lane, in_batch, L, B)
    m = tl.arange(0, L.mp)
    m_real = m < L.num_memory
    accumulate(grad_sum(acc[G_B2], L.num_memory, L.num_neuron, B) + m, tl.sum(dmo, axis=1), m_real)
    accumulate(grad_sum(acc[G_WY], L.num_memory, L.num_neuron, B) + m, g_wy, m_real)
    accumulate(grad_sum(acc[G_KM], L.num_memory, L.num_neuron, B) + m, tl.sum(dmd * mp, axis=1),
               m_real)  # fmt: skip
    accumulate(grad_sum(acc[G_KL], L.num_memory, L.num_neuron, B) + m, g_kl, m_real)
    b_ptr = grad_sum(acc[G_BIAS], 1, L.num_neuron, B)
    tl.store(b_ptr, tl.load(b_ptr) + g_bias)


########## GRADIENT WINDOW: MLP WEIGHT GRADIENTS ##########


@triton.jit
def window_step(bhist_ptr, mhist_ptr, kept, km, t0, s, n, lane, in_batch,
                L: tl.constexpr, B: tl.constexpr):  # fmt: skip
    """Step t0 + s of a window: the forward's branch sums and decayed memory and the backward's
    kept dmo."""
    t = t0 + s
    bact = tl.load(step(bhist_ptr, t, L.num_neuron * L.nbp + 1, B.stride)
                   + rows(n * L.nbp, L.nbp, lane, B.stride),
                   mask=(tl.arange(0, L.nbp) < L.num_branch)[:, None] & in_batch[None, :],
                   other=0.0).to(tl.float32)  # fmt: skip
    mp = tl.load(step(mhist_ptr, t, L.num_neuron * L.mp, B.stride)
                 + rows(n * L.mp, L.mp, lane, B.stride),
                 mask=(tl.arange(0, L.mp) < L.num_memory)[:, None] & in_batch[None, :],
                 other=0.0)  # fmt: skip
    md = km[:, None] * mp
    dmo = kept_rows(kept[K_DMO], s, n * L.mp, L.mp, L.num_neuron * L.mp, lane, B)
    return bact, md, dmo


@triton.jit
def elm_seq_mlp_wgrad(
    t0_ptr, bhist_ptr, mhist_ptr, kept, weights,
    acc,
    L: tl.constexpr, B: tl.constexpr, DOT: tl.constexpr,
):  # fmt: skip
    """Gradient window: one neuron's mlp weight gradients over the steps t0 .. t0 + window - 1,
    as the per-step backward would compute them, summed in its step order (latest first) and
    added to the sums."""
    t0 = tl.load(t0_ptr)
    n = tl.program_id(0)
    lane, in_batch = program_lanes(B)
    km = load_vector(weights[KM] + n * L.num_memory, L.mp, L.num_memory)
    if L.depth == 1:  # w2 (memory, hidden), w1b (hidden, branch), w1m (hidden, memory)
        for c in range(0, L.pieces):
            h0 = c * L.hc
            g_w2 = tl.zeros((L.mp, L.hc), tl.float32)
            g_w1b = tl.zeros((L.hc, L.nbp), tl.float32)
            g_w1m = tl.zeros((L.hc, L.mp), tl.float32)
            for i in range(0, L.window):
                s = L.window - 1 - i
                bact, md, dmo = window_step(bhist_ptr, mhist_ptr, kept, km, t0, s,
                                            n, lane, in_batch, L, B)  # fmt: skip
                h = kept_rows(kept[K_H], s, n * L.hp + h0, L.hc, L.num_neuron * L.hp, lane, B)
                dhpre = kept_rows(kept[K_DH], s, n * L.hp + h0, L.hc, L.num_neuron * L.hp,
                                  lane, B)  # fmt: skip
                g_w2 += dot(dmo, tl.trans(h), DOT)
                g_w1b += dot(dhpre, tl.trans(bact), DOT)
                g_w1m += dot(dhpre, tl.trans(md), DOT)
            accumulate_tile(grad_sum(acc[G_W2], L.num_memory * L.num_hidden, L.num_neuron, B),
                            g_w2, L.mp, L.hc, L.num_memory, L.num_hidden, 0, h0)  # fmt: skip
            accumulate_tile(grad_sum(acc[G_W1B], L.num_hidden * L.num_branch, L.num_neuron, B),
                            g_w1b, L.hc, L.nbp, L.num_hidden, L.num_branch, h0)  # fmt: skip
            accumulate_tile(grad_sum(acc[G_W1M], L.num_hidden * L.num_memory, L.num_neuron, B),
                            g_w1m, L.hc, L.mp, L.num_hidden, L.num_memory, h0)  # fmt: skip
    else:  # w2b (memory, branch), w2m (memory, memory)
        g_w2b = tl.zeros((L.mp, L.nbp), tl.float32)
        g_w2m = tl.zeros((L.mp, L.mp), tl.float32)
        for i in range(0, L.window):
            s = L.window - 1 - i
            bact, md, dmo = window_step(bhist_ptr, mhist_ptr, kept, km, t0, s, n,
                                        lane, in_batch, L, B)  # fmt: skip
            g_w2b += dot(dmo, tl.trans(bact), DOT)
            g_w2m += dot(dmo, tl.trans(md), DOT)
        accumulate_tile(grad_sum(acc[G_W2B], L.num_memory * L.num_branch, L.num_neuron, B),
                        g_w2b, L.mp, L.nbp, L.num_memory, L.num_branch)  # fmt: skip
        accumulate_tile(grad_sum(acc[G_W2M], L.num_memory * L.num_memory, L.num_neuron, B),
                        g_w2m, L.mp, L.mp, L.num_memory, L.num_memory)  # fmt: skip
