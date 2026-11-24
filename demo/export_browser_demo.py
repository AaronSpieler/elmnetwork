"""Export a trained ELMNetwork checkpoint to an ELMD v3 file for demo/index.html.

    python demo/export_browser_demo.py <run_dir> [--out-dir DIR]

`<run_dir>` can be either:
  - a training run folder containing `artifacts/` (with `main_config.json`, `model_config.json`,
    and `best_model_state_dict.eqx`), or
  - a standalone folder (such as `analysis/`) containing `model_config.json` and
    `best_model_state_dict.eqx` (pass `--dataset-dir` to locate the dataset cache).

Writes `elm_model.bin` (large weights in float16, deduplicated branch pools, byte
vocabulary, and float16 reference logits for the browser self-test) and copies
`demo/index.html` alongside it.
"""

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # demo/
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

PAGE = HERE / "index.html"
MAGIC, VERSION = b"ELMD", 3
DTYPES = {"float32": "f32", "uint32": "u32", "int32": "i32"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dir", help="training run folder or folder containing model_config.json")
    p.add_argument("--out-dir", help="output folder (default <artifacts>/browser_demo)")
    p.add_argument("--out-name", default="elm_model.bin", help="output .bin filename")
    p.add_argument("--checkpoint", default="best_model_state_dict.eqx")
    p.add_argument("--dataset", default=None, help="dataset name (default from config, else enwik8)")
    p.add_argument("--dataset-dir", default=None, help="folder containing dataset cache.pt")
    p.add_argument("--name", default=None, help="display name stored in header")
    p.add_argument("--test-steps", type=int, default=200, help="self-test characters")
    return p.parse_args()


def resolve_paths(run_dir: Path, checkpoint: str):
    artifacts = run_dir / "artifacts" if (run_dir / "artifacts").is_dir() else run_dir
    for fname in ("model_config.json", checkpoint):
        if not (artifacts / fname).is_file():
            raise SystemExit(f"{artifacts / fname} not found")
    cfg = {}
    main_json = artifacts / "main_config.json"
    exp_yaml = run_dir / "curr_experiment_config.yaml"
    if main_json.is_file():
        cfg = json.loads(main_json.read_text())
    elif exp_yaml.is_file():
        from omegaconf import OmegaConf

        cfg = OmegaConf.to_container(OmegaConf.load(exp_yaml), resolve=True)
    return artifacts, cfg


def load_model(artifacts: Path, checkpoint: str):
    import equinox as eqx
    import jax.random as jrandom

    from src.models.elm_network import ELMNetwork

    config = json.loads((artifacts / "model_config.json").read_text())
    config["fused"] = False  # exact float32 reference on plain JAX
    model = ELMNetwork(**config, key=jrandom.PRNGKey(0))
    return eqx.tree_deserialise_leaves(str(artifacts / checkpoint), model)


def load_vocab_and_test(cfg: dict, dataset: str | None, dataset_dir: str | None, steps: int):
    from src.datasets.language.data_loader import get_lm_corpus

    name = dataset or cfg.get("data", {}).get("dataset_name") or "enwik8"
    if dataset_dir is not None:
        folder = Path(os.path.expandvars(os.path.expanduser(dataset_dir)))
    else:
        base = cfg.get("setup", {}).get("datasets_base_folder")
        if not base:
            raise SystemExit("pass --dataset-dir (no datasets_base_folder found in config)")
        folder = Path(os.path.expandvars(os.path.expanduser(str(base)))) / "language_modeling" / name
    corpus = get_lm_corpus(str(folder), name)
    symbols = [str(s) for s in corpus.vocab.idx2sym]
    test_tokens = np.asarray(corpus.test[:steps]).astype(np.int32).reshape(-1)
    return name, symbols, test_tokens


def byte_of_index(symbols: list) -> list[int]:
    if not all(s.isdigit() and int(s) < 256 for s in symbols):
        raise SystemExit("unsupported vocabulary: the demo requires byte symbols (e.g. enwik8)")
    values = [int(s) for s in symbols]
    if len(set(values)) != len(values):
        raise SystemExit("unsupported vocabulary: duplicate byte symbols")
    return values


def check_model(model, vocab_size: int):
    from src.models import fused_layer as fk

    if model.input_embedding != "one_hot":
        raise SystemExit("unsupported model: demo requires one_hot input embedding")
    if model.num_input != vocab_size:
        raise SystemExit(f"input size {model.num_input} != vocabulary size {vocab_size}")
    if model.output_embedding != "linear_layer":
        raise SystemExit("unsupported model: demo requires linear_layer output embedding")
    for idx, layer in enumerate(model.layers):
        fk.check_fused_support(layer)
        spec = fk.layer_fused_spec(layer)
        if spec.activation not in ("identity", "relu", "spike"):
            raise SystemExit(f"unsupported layer {idx} activation: {spec.activation}")
        if spec.mlp_depth == 1 and spec.mlp_act not in ("relu", "relu_squared", "silu"):
            raise SystemExit(f"unsupported layer {idx} mlp_act: {spec.mlp_act}")
        conn = np.asarray(layer.input_connectome)
        if conn.min() < 0 or conn.max() >= layer.num_input + layer.num_neuron:
            raise SystemExit(f"layer {idx}: connectome index out of range")
        if idx + 1 < len(model.layers) and model.layers[idx + 1].num_input != layer.num_neuron:
            raise SystemExit(f"layer {idx + 1} input size != layer {idx} num_neuron")


def round_model_f16(model):
    """Return a copy of `model` with large synapse & MLP weights rounded through float16."""
    import equinox as eqx
    import jax.numpy as jnp

    def to_f16(x):
        return jnp.asarray(np.asarray(x, np.float32).astype(np.float16).astype(np.float32))

    new_layers = []
    for layer in model.layers:
        ens = layer.ensemble
        ens = eqx.tree_at(lambda e: e._w_s, ens, to_f16(ens._w_s))
        for i in range(len(ens.mlp.layers)):
            ens = eqx.tree_at(lambda e, k=i: e.mlp.layers[k].weight, ens, to_f16(ens.mlp.layers[i].weight))
        new_layers.append(eqx.tree_at(lambda l: l.ensemble, layer, ens))
    return eqx.tree_at(lambda m: m.layers, model, new_layers)


def bpc(logits: np.ndarray, tokens: np.ndarray) -> float:
    lg = logits[:-1].astype(np.float64)
    lg = lg - lg.max(axis=-1, keepdims=True)
    logp = lg - np.log(np.exp(lg).sum(axis=-1, keepdims=True))
    return float(-logp[np.arange(len(lg)), tokens[1:]].mean() / np.log(2))


def f16_words(a: np.ndarray) -> np.ndarray:
    """Pack pairs along the last axis as float16 into uint32 (low half first)."""
    if a.shape[-1] % 2:
        a = np.concatenate([a, np.zeros(a.shape[:-1] + (1,), a.dtype)], axis=-1)
    h = np.asarray(a, np.float32).astype(np.float16).view(np.uint16).astype(np.uint32)
    return h[..., 0::2] | (h[..., 1::2] << 16)


def pack_wiring(conn: np.ndarray, n: int, nb: int, ps: int):
    """Deduplicate per-branch source pools into (pool_map (n, nb), pools (num_pools, ps/2) u32)."""
    c = np.asarray(conn).reshape(n, nb, ps)
    if ps % 2:
        c = np.concatenate([c, np.zeros((n, nb, 1), c.dtype)], axis=-1)
    pool_map, pools, offset = np.zeros((n, nb), np.uint32), [], 0
    for b in range(nb):
        uniq, inv = np.unique(c[:, b, :], axis=0, return_inverse=True)
        pool_map[:, b] = inv.reshape(-1) + offset
        pools.append(uniq)
        offset += len(uniq)
    pools = np.concatenate(pools).astype(np.uint32)
    if pools.max() >= 1 << 16:
        raise SystemExit("unsupported model: more than 65,536 input slots per layer")
    return pool_map, pools[:, 0::2] | (pools[:, 1::2] << 16)


class Blob:
    """16-byte aligned little-endian array table."""

    def __init__(self):
        self.data, self.size, self.arrays = {}, 0, {}

    def add(self, name: str, a, dtype):
        a = np.ascontiguousarray(np.asarray(a), dtype=np.dtype(dtype).newbyteorder("<"))
        self.size += (16 - self.size % 16) % 16
        self.arrays[name] = [DTYPES[np.dtype(dtype).name], self.size, int(a.size)]
        self.data[name] = a
        self.size += a.nbytes


def pack_layers(model_f16, blob: Blob) -> list[dict]:
    from src.models import fused_layer as fk

    metas = []
    for idx, layer in enumerate(model_f16.layers):
        spec = fk.layer_fused_spec(layer)
        p = fk.layer_step_params(layer)
        n, nin = spec.num_neuron, spec.num_input
        nb, ps, m = spec.num_branch, spec.num_syn, spec.num_memory
        npy = {k: np.asarray(v, np.float32) for k, v in p.items() if k != "conn"}
        prefix = f"L{idx}."

        # synapses: (neuron, syn_pair, branch)
        blob.add(prefix + "ws", f16_words(npy["w_s"].reshape(n, nb, ps)).transpose(0, 2, 1), np.uint32)
        pool_map, pools = pack_wiring(p["conn"], n, nb, ps)
        blob.add(prefix + "pool_map", pool_map, np.uint32)
        blob.add(prefix + "pools", pools, np.uint32)

        # MLP: (neuron, input_pair, out_unit)
        if spec.mlp_depth == 1:
            h = spec.mlp_hidden
            blob.add(prefix + "mlp1", f16_words(npy["w1"]).transpose(0, 2, 1), np.uint32)
            blob.add(prefix + "mlp2", f16_words(npy["w2"]).transpose(0, 2, 1), np.uint32)
            parts = {
                "km": npy["km"], "kl": npy["kl"], "bias": npy["bias"],
                "wy": npy["wy"].reshape(n, m), "b1": npy["b1"], "b2": npy["b2"],
            }
        else:
            h = 0
            blob.add(prefix + "mlp1", f16_words(npy["w0"]).transpose(0, 2, 1), np.uint32)
            parts = {
                "km": npy["km"], "kl": npy["kl"], "bias": npy["bias"],
                "wy": npy["wy"].reshape(n, m), "b0": npy["b0"],
            }

        offsets, flat = {}, []
        for key, value in parts.items():
            offsets[key] = sum(len(f) for f in flat)
            flat.append(np.asarray(value, np.float32).reshape(-1))
        blob.add(prefix + "P", np.concatenate(flat), np.float32)

        # initial state: memory (n, m) + optional high-pass state (n)
        s0_len = n * (m + 1) if spec.high_pass else n * m
        blob.add(prefix + "S0", np.zeros(s0_len, np.float32), np.float32)
        blob.add(prefix + "a0", np.zeros(n, np.float32), np.float32)

        metas.append({
            "num_neuron": n,
            "num_input": nin,
            "num_branch": nb,
            "num_syn": ps,
            "syn_pairs": (ps + 1) // 2,
            "num_memory": m,
            "num_hidden": h,
            "mlp_depth": int(spec.mlp_depth),
            "mlp_act": spec.mlp_act,
            "gain": float(spec.gain),
            "activation": spec.activation,
            "high_pass": bool(spec.high_pass),
            "kappa_r": float(spec.kappa_r),
            "num_pools": int(pools.shape[0]),
            "p": offsets,
        })
    return metas


def check_round_trip(model_f16, blob: Blob, metas: list[dict]):
    """Verify packed f16 weights and wiring unpack back to exact float16 values."""
    from src.models import fused_layer as fk

    def f16(words):
        h = np.stack([words & 0xFFFF, words >> 16], -1).astype(np.uint16)
        return h.view(np.float16).astype(np.float32).reshape(*words.shape[:-1], -1)

    for idx, (layer, meta) in enumerate(zip(model_f16.layers, metas)):
        p = fk.layer_step_params(layer)
        n, nb, ps, m = meta["num_neuron"], meta["num_branch"], meta["num_syn"], meta["num_memory"]
        ps2, prefix = meta["syn_pairs"], f"L{idx}."
        ws = f16(blob.data[prefix + "ws"].reshape(n, ps2, nb).transpose(0, 2, 1))[..., :ps]
        pools = blob.data[prefix + "pools"].reshape(-1, ps2)
        idx_pairs = np.stack([pools & 0xFFFF, pools >> 16], -1).reshape(len(pools), -1)
        conn = idx_pairs[blob.data[prefix + "pool_map"].reshape(n, nb)][..., :ps]
        checks = [
            (ws, np.asarray(p["w_s"]).reshape(n, nb, ps)),
            (conn, np.asarray(p["conn"]).reshape(n, nb, ps)),
        ]
        k = nb + m
        if meta["mlp_depth"] == 1:
            h = meta["num_hidden"]
            w1 = f16(blob.data[prefix + "mlp1"].reshape(n, (k + 1) // 2, h).transpose(0, 2, 1))[..., :k]
            w2 = f16(blob.data[prefix + "mlp2"].reshape(n, (h + 1) // 2, m).transpose(0, 2, 1))[..., :h]
            checks += [(w1, np.asarray(p["w1"])), (w2, np.asarray(p["w2"]))]
        else:
            w0 = f16(blob.data[prefix + "mlp1"].reshape(n, (k + 1) // 2, m).transpose(0, 2, 1))[..., :k]
            checks += [(w0, np.asarray(p["w0"]))]
        for got, want in checks:
            if not np.array_equal(np.asarray(got, np.float32).reshape(-1), np.asarray(want, np.float32).reshape(-1)):
                raise SystemExit(f"layer {idx}: packed arrays did not round-trip identically")


def export_model(model, bytes_of_tokens: list[int], test_tokens: np.ndarray, *, name: str, dataset: str, test_bpc: float | None):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom

    vocab_size = len(bytes_of_tokens)
    check_model(model, vocab_size)

    logits32 = np.asarray(model(x=jnp.asarray(test_tokens), key=jrandom.PRNGKey(0), inference=True)[0], np.float32)
    if not np.all(np.isfinite(logits32)):
        raise SystemExit("reference float32 logits are not finite")

    model_f16 = round_model_f16(model)
    logits16 = np.asarray(model_f16(x=jnp.asarray(test_tokens), key=jrandom.PRNGKey(0), inference=True)[0], np.float32)

    blob = Blob()
    metas = pack_layers(model_f16, blob)
    out = model.linear_embedding
    n_last = model.layers[-1].num_neuron
    out_w = np.asarray(out.weight, np.float32).reshape(vocab_size, n_last)
    out_b = np.asarray(out.bias if out.bias is not None else np.zeros(vocab_size), np.float32)
    blob.add("out", np.concatenate([out_w.T.reshape(-1), out_b]), np.float32)
    blob.add("test.tokens", test_tokens, np.uint32)
    blob.add("test.logits", logits16, np.float32)
    check_round_trip(model_f16, blob, metas)

    trainable = eqx.filter(model, model.get_train_params_filter())
    num_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(trainable))
    header = {
        "format": VERSION,
        "name": name,
        "dataset": dataset,
        "trained_parameters": num_params,
        "test_bpc": float(test_bpc) if test_bpc is not None else None,
        "vocab_size": vocab_size,
        "input_scale": float(model.input_embedding_args.get("scale", 1.0)),
        "byte_of_index": bytes_of_tokens,
        "layers": metas,
        "test": {
            "steps": int(len(test_tokens)),
            "bpc": bpc(logits16, test_tokens),
            "bpc_float32": bpc(logits32, test_tokens),
        },
        "arrays": blob.arrays,
    }
    return header, blob


def write_model(path: Path, header: dict, blob: Blob):
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = MAGIC + struct.pack("<II", VERSION, len(head)) + head
    prefix += b"\0" * ((16 - len(prefix) % 16) % 16)
    with open(path, "wb") as f:
        f.write(prefix)
        written = 0
        for arr_name, arr in blob.data.items():
            f.write(b"\0" * (blob.arrays[arr_name][1] - written))
            f.write(memoryview(arr).cast("B"))
            written = blob.arrays[arr_name][1] + arr.nbytes


def main():
    args = parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")  # exact float32 reference logits
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")

    run_dir = Path(args.run_dir)
    artifacts, cfg = resolve_paths(run_dir, args.checkpoint)
    model = load_model(artifacts, args.checkpoint)
    dataset, symbols, test_tokens = load_vocab_and_test(cfg, args.dataset, args.dataset_dir, args.test_steps)

    test_bpc = None
    eval_path = artifacts / "eval_results.json"
    if eval_path.is_file():
        test_bpc = json.loads(eval_path.read_text()).get("final_test_bpc")
    display_name = args.name or f"ELM Network ({dataset})"

    header, blob = export_model(
        model, byte_of_index(symbols), test_tokens, name=display_name, dataset=dataset, test_bpc=test_bpc
    )
    out_dir = Path(args.out_dir) if args.out_dir else artifacts / "browser_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / args.out_name
    write_model(out_file, header, blob)
    shutil.copyfile(PAGE, out_dir / "index.html")
    size_mb = out_file.stat().st_size / 1e6
    test = header["test"]
    print(f"wrote {out_file} ({size_mb:.1f} MB) and {out_dir / 'index.html'}")
    print(f"  {header['name']} · {header['trained_parameters']:,} trained parameters")
    print(f"  float16 self-test ({test['steps']} chars): {test['bpc']:.4f} bpc (float32: {test['bpc_float32']:.4f})")


if __name__ == "__main__":
    main()
