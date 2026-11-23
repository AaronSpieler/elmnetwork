"""Times one training step of an ELM network on a GPU and reports median time, peak memory, and loss.

One warm-up step, then the median of --repeats steps. Run from the repository root:

    python src/profiling/bench_step.py --mode fused --num-hidden 4096 --num-memory 32 \
        --batch 64 --steps 100

Compare plain JAX, branch-pool shared lists, and fused kernels via --mode:
  - plain  : random wiring, plain JAX (with per-step gradient checkpointing)
  - pooled : random_pooled wiring, plain-JAX shared lists (with gradient checkpointing)
  - fused  : fused=True Triton GPU kernels (pooled or random wiring)
"""

import argparse
import os
import statistics
import sys
import time
import warnings

VOCAB = 204
REG_CONFIG = {
    "neuron_mlp_magnitude": {"layers": [0], "strength": 0.01, "proportional": False},
    "population_activity": {"layers": [0], "strength": 1.0, "proportional": False},
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--mode",
        choices=("plain", "pooled", "fused"),
        default="fused",
        help="execution path to benchmark",
    )
    p.add_argument("--num-hidden", type=int, default=4096, help="neurons per hidden layer")
    p.add_argument("--num-memory", type=int, default=32, help="memory units d_m")
    p.add_argument("--num-branch", type=int, default=None, help="branches per neuron (default: 4 * d_m)")
    p.add_argument(
        "--num-synapse",
        type=int,
        default=None,
        help="synapses per branch (default: 2 * d_m)",
    )
    p.add_argument(
        "--share-size",
        type=int,
        default=64,
        help="neurons per shared input list (pooled wiring)",
    )
    p.add_argument(
        "--wiring",
        choices=("pooled", "random"),
        default=None,
        help="wiring override (default: random for --mode plain, pooled otherwise)",
    )
    p.add_argument("--max-lanes", type=int, default=64, help="fused: max batch lanes per program")
    p.add_argument(
        "--hidden-chunk",
        type=int,
        default=None,
        help="fused: MLP hidden rows per tile (power of two >= 8; default: auto, max 64)",
    )
    p.add_argument(
        "--fast-backward",
        type=int,
        default=10,
        help="fused + pooled: gradient accumulation window (must divide --steps)",
    )
    p.add_argument("--precision", default="BF16_BF16_F32", help="jax_default_matmul_precision")
    p.add_argument("--batch", type=int, default=64, help="batch size")
    p.add_argument("--steps", type=int, default=100, help="time steps per sequence")
    p.add_argument("--repeats", type=int, default=5, help="timed repeats after warm-up")
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="disable per-step gradient checkpointing on plain-JAX modes",
    )
    return p


def build_step(args):
    root = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    import equinox as eqx
    import jax
    import jax.random as jrandom

    from src.datasets.language.language_train_utils import language_loss_and_grads
    from src.models.elm_network import ELMNetwork
    from src.training.regularizer import ELMNetworkRegularizer

    dm = args.num_memory
    nb = args.num_branch if args.num_branch is not None else 4 * dm
    nsyn = args.num_synapse if args.num_synapse is not None else 2 * dm
    wiring = args.wiring or ("random" if args.mode == "plain" else "pooled")
    fused = args.mode == "fused"
    grad_checkpointing = not args.no_checkpoint and not fused

    layer_config = {
        "input_wiring": "random_pooled" if wiring == "pooled" else "random",
        "input_wiring_args": (
            {
                "recurrent_connect": 0.8,
                "share_size": args.share_size,
            }
            if wiring == "pooled"
            else {"recurrent_connect": 0.8}
        ),
        "neuron_config": {
            "input_scale": 10.0,
            "num_branch": nb,
            "num_synapse_per_branch": nsyn,
            "num_memory": dm,
            "mlp_num_hidden_units": 2 * dm,
            "mlp_hidden_act": "relu_squared",
            "memory_tau_min": 0.1,
            "memory_tau_max": 100.0,
            "high_pass_tau": 2.0,
        },
    }
    if fused:
        fast_bwd = args.fast_backward if wiring == "pooled" else 1
        hidden_chunk = (
            args.hidden_chunk
            if args.hidden_chunk is not None
            else min(64, max(8, 1 << (2 * dm - 1).bit_length()))
        )
        layer_config["fused_kernels_args"] = {
            "interpret": False,
            "fast_backward": fast_bwd,
            "max_lanes": args.max_lanes,
            "hidden_chunk": hidden_chunk,
        }

    net = ELMNetwork(
        num_input=VOCAB,
        num_hidden=args.num_hidden,
        num_output=VOCAB,
        input_embedding="one_hot",
        input_embedding_args={"scale": 3.0},
        output_embedding="linear_layer",
        layer_config=layer_config,
        last_layer_override={"neuron_config": {"num_memory": 3, "mlp_num_hidden_layer": 0}},
        fused=fused,
        key=jrandom.PRNGKey(0),
    )
    diff, stat = eqx.partition(net, net.get_train_params_filter())
    reg = ELMNetworkRegularizer(REG_CONFIG)
    x = jrandom.randint(jrandom.PRNGKey(1), (args.batch, args.steps), 0, VOCAB)
    y = jrandom.randint(jrandom.PRNGKey(2), (args.batch, args.steps), 0, VOCAB)
    f = jax.jit(
        lambda d: language_loss_and_grads(
            d,
            stat,
            x,
            y,
            jrandom.PRNGKey(3),
            args.batch,
            VOCAB,
            0,
            1e-4,
            None,
            False,
            True,
            reg,
            grad_checkpointing,
        )
    )
    return f, diff


def main():
    warnings.filterwarnings("ignore")
    args = parser().parse_args()
    import jax

    assert jax.default_backend() == "gpu", "run this on a GPU node"
    with jax.default_matmul_precision(args.precision):
        f, diff = build_step(args)
        compiled = f.lower(diff).compile()
        mem = compiled.memory_analysis()
        peak_mb = (
            (
                mem.temp_size_in_bytes
                + mem.argument_size_in_bytes
                + mem.output_size_in_bytes
            )
            / (1024**2)
            if mem is not None
            else float("nan")
        )
        out = jax.block_until_ready(compiled(diff))  # warm-up
        times = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            out = jax.block_until_ready(compiled(diff))
            times.append(time.perf_counter() - t0)
        loss = float(out[0][0])
    summary = " ".join(f"{k}={v}" for k, v in vars(args).items())
    print(
        f"RESULT {summary} ms={statistics.median(times) * 1e3:.1f} "
        f"peak_mb={peak_mb:.1f} loss={loss!r}"
    )


if __name__ == "__main__":
    main()
