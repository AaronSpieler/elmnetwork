
# Scaling Laws and Tradeoffs in Recurrent Networks of Expressive Neurons

This repository provides an implementation of the ELM Network in Jax with Equinox, including minimal training scripts and experiment launch orchestration code.

![The ELM Neuron and Network](./elm_neuron_network.jpeg)

## Launching Experiments

***Prior to launching an experiment you need to:***

1) download the dataset e.g. Enwik8 (see below)
2) install the conda environment e.g. `elmnetwork_env.yml` (see below)
3) define your setup (paths, envs, wandb etc.) in e.g. `configs/setup/local_default.yaml`
4) define or modify your experiments e.g. `experiments/number_vs_complexity_of_neuron_enwik8.yaml`

make sure your experiment definition uses the correct default configs from `./configs` e.g. your setup yaml.

***Launching one of the provided / modified experiments:***

From the package root folder, run in the terminal:

```
conda activate elmnetwork_env
python src/training/local_launch_script.py -cn=number_vs_complexity_of_neuron_enwik8
```

for a Slurm cluster with Nvidia GPUs use `elmnetwork_env_gpu` with `launch_script`.

***Quick debug run, analysis, modifications:***

- if you want to overwrite existing model / training parameters simply modify the python call (e.g. for debug_run):
    - ``` python src/training/local_launch_script.py -cn=number_vs_complexity_of_neuron_enwik8 setup.debug_run=True```
- if you use wandb you may easily retrieve the results of a sweep check out `analysis/experiment_plotting.ipynb` as an example.
- if you want to write custom training code you may take a look at `src/training/training_script_enwik8.py` as a start.

## Repository Overview

***The core model and training code is in:***
```
src/
├── datasets/
├── models/
└── training/
```
- the ELM Neuron and Network implementations are in `src/models`
- the dataset specific dataloaders, loss functions and plotting functions are in `src/datasets`
- the training scripts and local and cluster launch scripts are in `src/training`

***The experiment and sweep configurations are in:***
```
configs/
experiments/
```

- the specific default model, training or setup `.yaml` configs are here
- you MUST adjust `configs/setup` to your environment to launch experiments
- full experiment definitions including multi-seed and parameters sweeps are here

***Some experiment and model analysis code is in:***
```
analysis/
```
- example trained model, and notebook to load model and run inference
- additional notebook to retrieve experiments from wandb (some provided) and simple plotting
- // potentially power-law fitting related analysis code will be added later

***The browser demo is in:***
```
demo/
```
- interactive inference page (`index.html`) and checkpoint exporter (`export_browser_demo.py`)

## Environment Setup

For running experiments on an Nvidia GPU cluster please use the CUDA version:

```
conda env create -f elmnetwork_env_gpu.yml
```

For CPU-only development use `elmnetwork_env.yml`.

if solving is slow or the environment seems conflicted make sure to enable:

```
conda config --set solver libmamba
conda config --set channel_priority flexible
```

For instructions on how to install miniconda please refer to: [installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)

## GPU Acceleration

For larger ELM networks on Nvidia GPUs, two model options provide major speedups:

***Branch pools (`input_wiring: "random_pooled"`):***

- pools of `share_size` branches (partitioned independently per branch slot across neurons) share input sources while keeping independent synapse weights (default `16`; `32`–`64` is faster on large layers)
- pair with `training.grad_checkpointing=True` (see `configs/model/elmnet_enwik8_pooled.yaml`)

***Fused Triton kernels (`fused: True`):***

- runs the recurrent sequence in custom Triton GPU kernels (see `configs/model/elmnet_enwik8_fused.yaml`)
- pair with `setup.matmul_precision: "BF16_BF16_F32"` on Ampere+ GPUs
- optional per-layer `fused_kernels_args` (mainly `fast_backward`, `max_lanes`, `hidden_chunk`; see `src/models/fused_layer.py`)
- note: the fused Triton kernel code was written fully LLM-assisted

***Benchmarking on your GPU:***

- use `src/profiling/bench_step.py` to time single training steps across modes (`--mode`)
- benchmark kernel options via `--share-size`, `--fast-backward`, `--max-lanes`, `--hidden-chunk`

```
python src/profiling/bench_step.py --mode fused --num-hidden 4096 --num-memory 32 --batch 64 --steps 100
```

Per-step times on one NVIDIA A100-40GB (4096 neurons, 32 memory units, batch size 64, 100 steps):

| Configuration | Step time | Speedup |
|---|---:|---:|
| Plain JAX, random wiring (TF32, checkpointed) | 12,103 ms | 1.0x |
| + Branch pools (`share_size=64`) | 1,105 ms | 11.0x |
| + Fused Triton kernels (`fast_backward=10`, `BF16_BF16_F32`) | 208 ms | 58.2x |

## Browser Demo

Try trained ELM networks live in your browser — character-by-character text generation with live hidden-layer neuron activity, running locally on your CPU or GPU (if available):

- live demo: [aaronspieler.github.io/elmnetwork](https://aaronspieler.github.io/elmnetwork/)
- pretrained models: [huggingface.co/AaronSpieler/elmnetwork](https://huggingface.co/AaronSpieler/elmnetwork) (selectable directly in the demo, or drop a local `.bin` file)
- note: the browser demo code was written fully LLM-assisted

***Exporting a trained run:***

- export a character-level language-modeling run directory to a `.bin` file for the demo:

```
python demo/export_browser_demo.py <run_dir> [--out-dir DIR]
```

- supports standard ELM network configs (except `branch_tau` and `mlp_num_hidden_layer > 1`)

## Dataset Setup

The training scripts expect the following data structure overall

```
<DATASETS BASE FOLDER>/
├── language_modeling/
│   └── enwik8/
├── heidelberg/
└── neuronio/
    ├── train/
    └── test/
        └── Data_test/
```

you only need to download the specific dataset you want to run experiments on.

***Preparing the Enwik8 Dataset:***

You can simply run the following commands to download the dataset (~1.3GB):

```
mkdir -p <DATASETS BASE FOLDER>/language_modeling/enwik8
cd <DATASETS BASE FOLDER>/language_modeling/enwik8
wget --continue http://mattmahoney.net/dc/enwik8.zip
wget https://raw.githubusercontent.com/salesforce/awd-lstm-lm/master/data/enwik8/prep_enwik8.py
python3 prep_enwik8.py
```

If you wish to download a whole suite of language datasets run:

```
bash src/datasets/language/getdata.sh
```

***Preparing the SHD and SHD-Adding Dataset:***

You may download the dataset using the provided utils (~0.5GB):

```
from src.datasets.shd.shd_download_utils import get_shd_dataset
get_shd_dataset(<DATASETS BASE FOLDER>, "heidelberg")
```

Dataloaders for both datasets are provided.
More details can be found at the website: [spiking-heidelberg-datasets-shd](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)

***Preparing the NeuronIO Dataset:***

Running NeuronIO experiments (very optional) requires first downloading the dataset from Kaggle (~115GB).

- Train Data (train): [single-neurons-as-deep-nets-nmda-train-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-train-data)
- Test Data (test/Data_test): [single-neurons-as-deep-nets-nmda-test-data](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data)

More details can be found in the following repository: [neuron_as_deep_net](https://github.com/SelfishGene/neuron_as_deep_net)

## Citation

If you wish to cite us you can do so using

```
@inproceedings{spieler2026scaling,
  title={Scaling Laws and Tradeoffs in Recurrent Networks of Expressive Neurons},
  author={Spieler, Aaron and Martius, Georg and Levina, Anna},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={39},
  year={2026},
  url={https://openreview.net/forum?id=zpewdRsVpA}
}
```
