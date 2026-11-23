# --------------------------------------------------------------------------------
# Portions of the SHD dataloader in this file were adapted from the SpyTorch
# tutorial project by Friedemann Zenke, Manu Halvagal, and contributors.
# The original implementation is available at: https://github.com/fzenke/spytorch
# Licensed under CC BY 4.0; see src/licences/CC-BY-4.0.txt and
# https://creativecommons.org/licenses/by/4.0/
# This file has been modified from the original.
#
# The remaining code in this file, including the SHDAdding dataset wrapper,
# originates from: https://github.com/AaronSpieler/elmneuron/
# Licensed under the MIT License; see src/licences/elmneuron-MIT.txt.
# --------------------------------------------------------------------------------

import h5py
import numpy as np
import torch


def random_val_split_SHD_data(train_file, valid_fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    valid_condition = np.array(
        rng.binomial(1, valid_fraction, size=len(np.array(train_file["labels"]))),
        dtype=bool,
    )

    x_train = {
        "times": train_file["spikes"]["times"][~valid_condition],
        "units": train_file["spikes"]["units"][~valid_condition],
    }
    y_train = train_file["labels"][~valid_condition]

    x_valid = {
        "times": train_file["spikes"]["times"][valid_condition],
        "units": train_file["spikes"]["units"][valid_condition],
    }
    y_valid = train_file["labels"][valid_condition]

    return x_train, y_train, x_valid, y_valid


class SHD(torch.utils.data.IterableDataset):
    def __init__(
        self,
        X: h5py._hl.group.Group,
        y: h5py._hl.dataset.Dataset,
        batch_size: int,
        bin_size: int,
        shuffle: bool = False,
        test_set: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        self.num_input_channel = 700
        self.num_classes = 20
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.shuffle = shuffle
        self.test_set = test_set
        self.seed = seed

        # Load spiking data
        self.labels = np.array(y).astype(np.int32)
        self.firing_times = X["times"]
        self.units_fired = X["units"]

        # Check for test set
        if self.test_set:
            assert (
                len(self.labels) % batch_size == 0
            ), f"The dataset size {len(self.labels)} is not exactly divisible by batch size {batch_size}."

        # Epoch management
        self.rng = np.random.default_rng(seed)
        self.sample_index = np.arange(len(self.labels))
        self.batches_per_epoch = len(self.labels) // self.batch_size

        # Precompute time bins
        max_time, in_miliseconds = 1.0, 1000
        step_size = self.bin_size / in_miliseconds
        self.time_bins = np.arange(0, max_time + step_size, step_size)
        self.num_time_bins = len(self.time_bins) - 1

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.sample_index)

        raw_bins = len(self.time_bins)
        for counter in range(len(self)):
            batch_index = self.sample_index[
                self.batch_size * counter : self.batch_size * (counter + 1)
            ]

            coo = [[] for _ in range(3)]
            for batch_count, idx in enumerate(batch_index):
                times = np.clip(
                    np.digitize(self.firing_times[idx], self.time_bins) - 1,
                    0,
                    raw_bins - 1,
                )
                units = self.units_fired[idx]
                batch = [batch_count for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            # Create sparse matrix
            i = torch.tensor(coo, dtype=torch.long)
            v = torch.ones(len(coo[0]), dtype=torch.float32)
            X_batch = torch.sparse_coo_tensor(
                i,
                v,
                size=(self.batch_size, raw_bins, self.num_input_channel),
                dtype=torch.float32,
                check_invariants=False,
            ).to_dense()

            X_batch = X_batch[:, :-1]  # correct for spikes outside interval
            y_batch = torch.Tensor(self.labels[batch_index]).to(torch.int64)

            yield X_batch, y_batch


class SHDAdding(torch.utils.data.IterableDataset):
    """Our proposed SHD-Adding dataset, which consists of concatenated digits
    from the SHD dataset. The target is the sum of the digits.
    """

    def __init__(
        self,
        X: h5py._hl.group.Group,
        y: h5py._hl.dataset.Dataset,
        batch_size: int,
        bin_size: int,
        batches_per_epoch: int = 500,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()

        self.num_input_channel = 700
        self.num_classes = 19
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.shuffle = shuffle
        self.seed = seed

        # Load spiking data
        self.labels = np.array(y).astype(np.int32)
        self.firing_times = X["times"]
        self.units_fired = X["units"]

        # Epoch management
        self.rng = np.random.default_rng(self.seed)
        self.batches_per_epoch = batches_per_epoch

        # Precompute time bins
        max_time, in_miliseconds = 1.0, 1000
        step_size = self.bin_size / in_miliseconds
        self.time_bins = np.arange(0, max_time + step_size, step_size)
        self.num_time_bins = 2 * (len(self.time_bins) - 1)

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        if not self.shuffle:
            self.rng = np.random.default_rng(self.seed)

        raw_bins = len(self.time_bins)
        for i in range(self.batches_per_epoch):
            batch_index = self.rng.integers(
                low=0, high=len(self.labels), size=2 * self.batch_size
            )

            coo = [[] for _ in range(3)]
            for batch_count, idx in enumerate(batch_index):
                times = np.clip(
                    np.digitize(self.firing_times[idx], self.time_bins) - 1,
                    0,
                    raw_bins - 1,
                )
                units = self.units_fired[idx]
                batch = [batch_count for _ in range(len(times))]

                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.tensor(coo, dtype=torch.long)
            v = torch.ones(len(coo[0]), dtype=torch.float32)
            X_batch = torch.sparse_coo_tensor(
                i,
                v,
                size=(
                    2 * self.batch_size,
                    raw_bins,
                    self.num_input_channel,
                ),
                dtype=torch.float32,
                check_invariants=False,
            ).to_dense()

            y_batch = self.labels[batch_index]

            arrays_to_concatenate = [
                X_batch[
                    : self.batch_size, :-1, :
                ],  # correct for spikes outside interval
                X_batch[
                    self.batch_size :, :-1, :
                ],  # correct for spikes outside interval
            ]

            X_batch = torch.Tensor(np.concatenate(arrays_to_concatenate, axis=-2))

            # sum two numbers each (irrespective of english or german)
            # labels are 0-9 english, then 0-9 german
            y_batch = torch.Tensor(
                y_batch[: self.batch_size] % 10 + y_batch[self.batch_size :] % 10
            ).to(torch.int64)

            yield X_batch, y_batch
