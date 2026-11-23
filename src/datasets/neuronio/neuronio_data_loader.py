# --------------------------------------------------------------------------------
# This file contains portions adapted from the elmneuron project.
# The original implementation is available at: https://github.com/AaronSpieler/elmneuron/
# Copyright (c) 2023 Aaron Spieler.
# Licensed under the MIT License; see src/licences/elmneuron-MIT.txt.
# --------------------------------------------------------------------------------

import random
import time
from typing import List

import jax.numpy as jnp
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

from .neuronio_data_utils import (
    DEFAULT_Y_SOMA_THRESHOLD,
    DEFAULT_Y_TRAIN_SOMA_BIAS,
    DEFAULT_Y_TRAIN_SOMA_SCALE,
    NEURONIO_DATA_DIM,
    NEURONIO_LABEL_DIM,
    NEURONIO_SIM_LEN,
    NEURONIO_SIM_PER_FILE,
    create_neuronio_input_type,
    parse_sim_experiment_file,
)

"""
Code based on the SimulationDataGenerator that was written by David Beniaguev and Oren Amsalem and originates
from https://github.com/SelfishGene/neuron_as_deep_net/blob/master/fit_CNN.py.
The dataloading was adapted to generate batches asynchronously.
This dataloader is highly nondeterministic for multiple workers, even when seeded.
"""


def preprocess_data(
    X,
    y_spike,
    y_soma,
    y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
):
    # Convert to torch tensors
    X = torch.from_numpy(X).permute(2, 1, 0)
    y_spike = torch.from_numpy(y_spike).T.unsqueeze(2)
    y_soma = torch.from_numpy(y_soma).T.unsqueeze(2)

    # Apply thresholding
    y_soma[y_soma > y_soma_threshold] = y_soma_threshold

    # Bias correction and scaling
    y_soma = (y_soma - y_train_soma_bias) * y_train_soma_scale

    return X, y_spike, y_soma


def generate_batch(
    X,
    y_spike,
    y_soma,
    synapse_types,
    generate_batch_rng,
    batch_size: int,
    input_window_size: int,
    ignore_time_from_start: int,
    neuronio_sim_per_file: int = NEURONIO_SIM_PER_FILE,
    neuronio_sim_len: int = NEURONIO_SIM_LEN,
    neuronio_label_dim: int = NEURONIO_LABEL_DIM,
    neuronio_data_dim: int = NEURONIO_DATA_DIM,
):
    # randomly sample simulations for current batch
    selected_sim_inds = generate_batch_rng.choice(
        neuronio_sim_per_file, size=batch_size, replace=False
    )

    # randomly sample timepoints for current batch
    selected_time_inds = (
        generate_batch_rng.choice(
            neuronio_sim_len - input_window_size - ignore_time_from_start,
            size=batch_size,
            replace=True,
        )
        + ignore_time_from_start
    )

    # Initialize batch tensors
    X_batch = torch.zeros(
        (batch_size, input_window_size, neuronio_data_dim), dtype=torch.bool
    )
    y_spike_batch = torch.zeros(
        (batch_size, input_window_size, neuronio_label_dim // 2), dtype=torch.bool
    )
    y_soma_batch = torch.zeros(
        (batch_size, input_window_size, neuronio_label_dim // 2), dtype=torch.float32
    )

    # Gather batch data
    for k, (sim_ind, time_ind) in enumerate(zip(selected_sim_inds, selected_time_inds)):
        X_batch[k] = X[sim_ind, time_ind : time_ind + input_window_size, :]
        y_spike_batch[k] = y_spike[sim_ind, time_ind : time_ind + input_window_size, :]
        y_soma_batch[k] = y_soma[sim_ind, time_ind : time_ind + input_window_size, :]

    # NOTE: conversion to float as late as possible to save memory

    # Apply synapse types and formatting
    X_batch = (X_batch * synapse_types).to(torch.float32)
    y_spike_batch = y_spike_batch.squeeze(-1).to(torch.int32)
    y_soma_batch = y_soma_batch.squeeze(-1).to(torch.float32)

    # Return the batch
    return X_batch, (y_spike_batch, y_soma_batch)


def worker_fn(
    seed,
    file_paths,
    batch_queue,
    file_load_fraction: float,
    batch_size: int,
    input_window_size: int,
    ignore_time_from_start: int,
    y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
    neuronio_sim_per_file: int = NEURONIO_SIM_PER_FILE,
    neuronio_sim_len: int = NEURONIO_SIM_LEN,
    neuronio_label_dim: int = NEURONIO_LABEL_DIM,
    neuronio_data_dim: int = NEURONIO_DATA_DIM,
    ignore_synapse_types: bool = False,
    verbose: bool = False,
    worker_idx: int = 0,
):
    # set count for this worker process
    torch.set_num_threads(1)

    # initialize random number generators
    select_file_rng = random.Random(seed)
    generate_batch_rng = np.random.default_rng(seed)

    # sleep so not all start loading at same time
    time.sleep(worker_idx * 3)

    # create synapse types
    synapse_types = torch.tensor(create_neuronio_input_type(), dtype=torch.float32)
    if ignore_synapse_types:
        synapse_types = torch.abs(synapse_types)

    # calculate number of batches per file
    max_batches_per_file = (neuronio_sim_per_file * neuronio_sim_len) / (
        batch_size * input_window_size
    )
    batches_per_file = int(file_load_fraction * max_batches_per_file)

    # Loop until there are no more file paths or an exit condition is triggered
    while True:
        # Randomly select a file path
        file_path = select_file_rng.choice(file_paths)

        # Parse the simulation experiment file
        X, y_spike, y_soma = parse_sim_experiment_file(
            sim_experiment_file=file_path,
            include_params=False,
            verbose=verbose,
        )

        # Preprocess the raw data
        X, y_spike, y_soma = preprocess_data(
            X=X,
            y_spike=y_spike,
            y_soma=y_soma,
            y_soma_threshold=y_soma_threshold,
            y_train_soma_bias=y_train_soma_bias,
            y_train_soma_scale=y_train_soma_scale,
        )

        # Generate and enqueue batches
        batch_count = 0
        while batch_count < batches_per_file:
            # Generate actual batch
            batch = generate_batch(
                X=X,
                y_spike=y_spike,
                y_soma=y_soma,
                synapse_types=synapse_types,
                generate_batch_rng=generate_batch_rng,
                batch_size=batch_size,
                input_window_size=input_window_size,
                ignore_time_from_start=ignore_time_from_start,
                neuronio_sim_per_file=neuronio_sim_per_file,
                neuronio_sim_len=neuronio_sim_len,
                neuronio_label_dim=neuronio_label_dim,
                neuronio_data_dim=neuronio_data_dim,
            )

            # Try to add the batch to the batch_queue
            batch_queue.put(batch, block=True, timeout=None)
            batch_count += 1


class NeuronIO(IterableDataset):
    def __init__(
        self,
        batches_per_epoch: int,
        file_paths: List[str],
        batch_size: int = 8,
        input_window_size: int = 500,
        file_load_fraction: float = 0.3,
        ignore_time_from_start: int = 500,
        num_workers: int = 5,
        num_prefetch_batch: int = 50,
        y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
        y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
        y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
        neuronio_sim_per_file: int = NEURONIO_SIM_PER_FILE,
        neuronio_sim_len: int = NEURONIO_SIM_LEN,
        neuronio_label_dim: int = NEURONIO_LABEL_DIM,
        neuronio_data_dim: int = NEURONIO_DATA_DIM,
        ignore_synapse_types: bool = False,
        seed: int = 0,
        verbose: bool = False,
        torch_data=False,
        torch_device="cpu",
    ):
        super().__init__()
        self.batches_per_epoch = batches_per_epoch
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.input_window_size = input_window_size
        self.file_load_fraction = file_load_fraction
        self.ignore_time_from_start = ignore_time_from_start
        self.num_workers = num_workers
        self.num_prefetch_batch = num_prefetch_batch
        self.y_train_soma_bias = y_train_soma_bias
        self.y_train_soma_scale = y_train_soma_scale
        self.y_soma_threshold = y_soma_threshold
        self.neuronio_sim_per_file = neuronio_sim_per_file
        self.neuronio_sim_len = neuronio_sim_len
        self.neuronio_label_dim = neuronio_label_dim
        self.neuronio_data_dim = neuronio_data_dim
        self.ignore_synapse_types = ignore_synapse_types
        self.seed = seed
        self.verbose = verbose
        self.torch_data = torch_data
        self.torch_device = torch_device

        mp.set_start_method("spawn", force=True)
        self.batch_queue: mp.Queue = mp.Queue(maxsize=self.num_prefetch_batch)
        self.workers = []
        for i in range(self.num_workers):
            worker_seed = seed + i if seed is not None else None
            worker = mp.Process(
                target=worker_fn,
                args=(
                    worker_seed,
                    self.file_paths,
                    self.batch_queue,
                    self.file_load_fraction,
                    self.batch_size,
                    self.input_window_size,
                    self.ignore_time_from_start,
                    self.y_soma_threshold,
                    self.y_train_soma_bias,
                    self.y_train_soma_scale,
                    self.neuronio_sim_per_file,
                    self.neuronio_sim_len,
                    self.neuronio_label_dim,
                    self.neuronio_data_dim,
                    self.ignore_synapse_types,
                    self.verbose,
                    i,
                ),
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

        self.local_batch_buffer: list = []
        for i in range(num_prefetch_batch):
            self.prefetch_next_batch()

    def prefetch_next_batch(self, return_batch=False):
        # retrieve batch from queue
        batch = self.batch_queue.get(block=True, timeout=None)
        X_batch, (y_spike_batch, y_soma_batch) = batch

        # put on appropriate device
        if self.torch_data:
            X_batch = X_batch.to(self.torch_device, non_blocking=True)
            y_spike_batch = y_spike_batch.to(self.torch_device, non_blocking=True)
            y_soma_batch = y_soma_batch.to(self.torch_device, non_blocking=True)
        else:
            # jax asarray by default non blocking
            X_batch = jnp.asarray(X_batch, dtype=jnp.float32)
            y_spike_batch = jnp.asarray(y_spike_batch, dtype=jnp.int32)
            y_soma_batch = jnp.asarray(y_soma_batch, dtype=jnp.float32)

        # put in local buffer
        batch = X_batch, (y_spike_batch, y_soma_batch)
        self.local_batch_buffer.append(batch)

        if return_batch:
            # return first batch in queue
            return self.local_batch_buffer.pop(0)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(len(self)):
            yield self.prefetch_next_batch(return_batch=True)

    def __del__(self):
        try:
            self.batch_queue.cancel_join_thread()
            self.batch_queue.close()
        finally:
            for worker in self.workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join()
