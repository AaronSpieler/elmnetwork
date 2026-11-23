"""Launching the fused Triton kernels.

Compiled on a GPU through jax-triton (`launch(..., interpret=False)`), or in Triton's interpreter
on the CPU (`interpret=True`, for tests: the same kernel code, run by Triton's emulator on
PyTorch CPU tensors through a JAX callback). Interpret mode needs TRITON_INTERPRET=1 set before
the kernel modules are imported (Triton decides per function, at definition, whether it is
interpreted; the test setup sets it).

Kernel signature convention (jax-triton's): read-only inputs, then the buffers the kernel updates
in place, then its fresh outputs; compile-time constants (tl.constexpr) as keywords. A buffer a
variant does not use is None, buffers that belong together are a tuple (see launch).
"""

import os
import sys
import threading

import jax
import jax.numpy as jnp
import numpy as np
import triton

from .. import jaxtriton as jt

# conda builds of Triton ship no ptxas-blackwell (conda-forge/triton-feedstock#67), so compiling for
# H100 and newer fails. Fall back to the environment's own ptxas, unless the user chose one.
_BUNDLED = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas-blackwell")
_PTXAS = os.path.join(sys.prefix, "bin", "ptxas")
if not os.path.exists(_BUNDLED) and os.path.exists(_PTXAS):
    os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", _PTXAS)

# Triton states each kernel's exact thread count (.reqntid); ptxas then gives a kernel that needs
# more than 255 registers only 32 and spills (2.4x slower training steps). A register limit of
# the kernel's own (Triton's maxnreg, .maxnreg in the PTX) lifts that cap (triton-lang/triton#12846).
MAXNREG = 255


def launch(kernel, *, name, grid, inputs, inplace=(), outputs=(), interpret, num_warps,
           constants=None):  # fmt: skip
    """Runs `kernel` over `grid` and returns (updated in-place buffers, fresh outputs).

    inputs, inplace: one entry per kernel parameter, in the kernel's order. An entry is an
    array, None (a buffer this variant of the kernel does not use: the kernel sees None and
    compiles without it), or a tuple of arrays (buffers that travel together; a None member
    becomes a read-only dummy, since jax-triton takes no None inside a tuple: the kernel must
    know from its constants not to touch it, and None comes back). inputs are read
    only (Python scalars are compile-time specialized: pass values that change between calls,
    like a time step, as arrays); inplace buffers are read and updated, and come back in the
    same structure. outputs: jax.ShapeDtypeStruct of arrays the kernel fully writes.
    constants: the kernel's tl.constexpr parameters (hashable: numbers, strings, frozen
    dataclasses)."""
    constants = dict(constants or {})
    dummy = jnp.zeros((1,), jnp.float32)  # stands for an unused member of a tuple of buffers
    inputs, outputs, given = [_fill(x, dummy) for x in inputs], list(outputs), list(inplace)
    inplace = [_fill(x, dummy) for x in given]
    if interpret:
        from triton.runtime.interpreter import InterpretedFunction

        if not isinstance(kernel, InterpretedFunction):
            raise RuntimeError(
                "interpret mode: set TRITON_INTERPRET=1 before importing the kernels"
            )
        updated, fresh = _interpret(kernel, grid, inputs, inplace, outputs, num_warps,
                                    constants)  # fmt: skip
        return _drop(updated, given), fresh
    # (a top-level None is not a leaf: it stays None; a dummy is read only: no Ref)
    refs = jax.tree_util.tree_map(lambda x: x if x is dummy else jax.new_ref(x), inplace)
    outs = jt.triton_call(
        *inputs, *refs, kernel=kernel, out_type=tuple(outputs), grid=grid, name=name,
        num_warps=num_warps, num_stages=1, backend_options={"maxnreg": MAXNREG}, **constants,
    )  # fmt: skip
    updated = jax.tree_util.tree_map(lambda r: r if r is dummy else r[...], refs)
    return _drop(updated, given), list(outs)


def _fill(x, dummy):
    """A kernel argument with the None members of a tuple replaced by the dummy."""
    return tuple(dummy if y is None else y for y in x) if isinstance(x, tuple) else x


def _drop(updated, given):
    """The updated buffers with the dummies taken out again (None, as they were given)."""
    return [tuple(None if g is None else u for u, g in zip(x, y)) if isinstance(x, tuple) else x
            for x, y in zip(updated, given)]  # fmt: skip


def kernel_signature(kernel, inputs, inplace, outputs, constants):
    """What `launch` compiles, for Triton's own compiler (the compile check): the kernel's
    signature, constants and pointer attributes (every pointer 16-byte aligned, as jax-triton
    declares them). Usage: triton.compile(ASTSource(kernel, *kernel_signature(...)), ...)."""
    types = {"float32": "*fp32", "bfloat16": "*bf16", "int32": "*i32"}
    signature, constexprs, attrs = {}, dict(constants or {}), {}
    dummy = np.zeros((1,), np.float32)
    inputs, inplace = [_fill(x, dummy) for x in inputs], [_fill(x, dummy) for x in inplace]

    def describe(x, path):  # path: (parameter index[, index in its tuple]), Triton's keys
        if x is None:
            constexprs[path if len(path) > 1 else names[path[0]]] = None
            return "constexpr"
        if isinstance(x, tuple):
            return tuple(describe(y, path + (i,)) for i, y in enumerate(x))
        attrs[path] = [["tt.divisibility", 16]]
        return types[np.dtype(x.dtype).name]

    params = [p for p in kernel.params if not p.is_constexpr]
    names = {p.num: p.name for p in params}
    for p, x in zip(params, list(inputs) + list(inplace) + list(outputs)):
        signature[p.name] = describe(x, (p.num,))
    signature.update({k: "constexpr" for k in constants or {}})
    return signature, constexprs, attrs


# Triton's interpreter keeps the running grid in one global; with several (CPU) devices, JAX runs
# callbacks concurrently, so interpreted launches take turns
_INTERPRETER_LOCK = threading.Lock()


def _interpret(kernel, grid, inputs, inplace, outputs, num_warps, constants):
    tree = jax.tree_util
    in_leaves, in_def = tree.tree_flatten(inputs)  # arrays and Python scalars; None dropped
    up_leaves, up_def = tree.tree_flatten(inplace)
    is_array = [isinstance(x, jax.Array) for x in in_leaves]
    shapes = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in up_leaves) + tuple(outputs)

    def host(*values):
        import torch  # interpret mode only (tests)

        values = iter(values)
        ins = [_to_torch(next(values)) if a else x for x, a in zip(in_leaves, is_array)]
        ups = [_to_torch(next(values)) for _ in up_leaves]
        # fresh outputs start as NaN (integers: -1), so a value the kernel fails to write
        # shows up in the tests instead of passing as a zero
        fresh = [_fresh(torch, s) for s in outputs]
        with _INTERPRETER_LOCK:
            kernel[grid](*in_def.unflatten(ins), *up_def.unflatten(ups), *fresh,
                         num_warps=num_warps, **constants)  # fmt: skip
        return tuple(_to_numpy(t) for t in ups + fresh)

    arrays = [x for x, a in zip(in_leaves, is_array) if a] + up_leaves
    res = list(jax.pure_callback(host, shapes, *arrays))
    return up_def.unflatten(res[: len(up_leaves)]), res[len(up_leaves) :]


def _to_torch(x):
    import torch

    x = np.asarray(x)
    if x.dtype.name == "bfloat16":  # numpy has no native bf16: move the bits
        return torch.from_numpy(x.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(x.copy())


def _to_numpy(t):
    import ml_dtypes
    import torch

    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16).view(ml_dtypes.bfloat16)
    return t.numpy()


def _fresh(torch, s):
    dtype = {"bfloat16": torch.bfloat16}.get(np.dtype(s.dtype).name) or getattr(
        torch, np.dtype(s.dtype).name
    )
    fill = float("nan") if dtype.is_floating_point else -1
    return torch.full(tuple(s.shape), fill, dtype=dtype)
