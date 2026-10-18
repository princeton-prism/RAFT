from abc import ABC, abstractmethod
import time
from typing import Any, Protocol, Collection
import os
import math

import numpy as np
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from tqdm import tqdm


class TimeLogger(ABC):
    @abstractmethod
    def __enter__(self):
        ...
    
    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        ...

    @abstractmethod
    def get_last_elapsed_time(self):
        ...

class CPUTimeLogger(TimeLogger):
    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._elapsed_time = time.perf_counter() - self._start_time

    def get_last_elapsed_time(self):
        return self._elapsed_time

class GPUTimeLogger(TimeLogger):
    def __enter__(self):
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._end_event.record()
        torch.cuda.synchronize()
        self._elapsed_time = self._start_event.elapsed_time(self._end_event) / 1000.0
        return self

    def get_last_elapsed_time(self):
        return self._elapsed_time

class StateDictProtocol(Protocol):
    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state_dict: dict[str, Any]):
        ...

class Checkpointer:
    def __init__(self, checkpoint_dir: str, ckpt_latest_only: bool = False, **modules: StateDictProtocol):
        self.checkpoint_dir = checkpoint_dir
        self._modules = modules
        self.ckpt_latest_only = ckpt_latest_only

    def save(self, step: int):
        ckpt = {
            "step": step,
            **{k: v.state_dict() for k, v in self._modules.items()}
        }
        save_path = os.path.join(self.checkpoint_dir, "ckpt_latest.pth" if self.ckpt_latest_only else f"ckpt_{step}.pth")
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(ckpt, save_path)

    def load(self, step: int | None = None):
        assert not self.ckpt_latest_only or step is None, "Cannot specify step to load when ckpt_latest_only is True"
        if step is None:
            if self.ckpt_latest_only:
                ckpt_fn = "ckpt_latest.pth"
                if not os.path.exists(os.path.join(self.checkpoint_dir, ckpt_fn)):
                    return 0
            else:
                steps = []
                for f in os.listdir(self.checkpoint_dir):
                    if f.startswith("ckpt_") and f.endswith(".pth"):
                        e = int(f[:-len(".pth")].split("_")[-1])
                        steps.append(e)
                if len(steps) == 0:
                    return 0
                step = max(steps)
                ckpt_fn = f"ckpt_{step}.pth"
        else:
            ckpt_fn = f"ckpt_{step}.pth"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_fn)
        ckpt = torch.load(ckpt_path, weights_only=True)
        for k, v in self._modules.items():
            v.load_state_dict(ckpt[k])
        return ckpt["step"]

def build_wandb_config(config: DictConfig):
    return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

def safe_div(a: int, b: int):
    if a % b != 0:
        raise ValueError(f"Cannot divide {a} by {b} evenly!")
    return a // b

def create_dist_dataloader(dataset: Dataset, rank: int, world_size: int, config: DictConfig):
    loader_kwargs = {
        "persistent_workers": True,
        "pin_memory": True,
        "batch_size": safe_div(config["train"]["dataloader"]["batch_size"], world_size),
        "num_workers": config["train"]["dataloader"]["num_workers"],
    }
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, sampler=sampler, **loader_kwargs)

def gather_info(info_to_gather: dict[Any, float], world_size: int, reduction: str = "mean"):
    gathered_infos: list[dict[Any, float]] = [None] * world_size
    dist.all_gather_object(gathered_infos, info_to_gather)
    if reduction == "mean":
        reduce_fn = np.mean
    elif reduction == "sum":
        reduce_fn = np.sum
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
    return {k: reduce_fn([d[k] for d in gathered_infos]) for k in info_to_gather}

def nested_dict_to_flat_dict(d: dict[str, Any], pfx: str = ""):
    flat_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flat_dict.update(nested_dict_to_flat_dict(v, pfx=f"{pfx}{k}/"))
        else:
            flat_dict[f"{pfx}{k}"] = v
    return flat_dict


class DistributedEpisodeAwareSampler(Sampler):
    """
    Merged implementation of torch.utils.data.distributed.DistributedSampler and lerobot.datasets.sampler.EpisodeAwareSampler
    """
    def __init__(
        self,
        episode_data_index: dict[str, list[torch.IntTensor]],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        episode_indices_to_use: Collection[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.unshuffled_indices: list[int] = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                self.unshuffled_indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.dataset_len = len(self.unshuffled_indices)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.dataset_len % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.dataset_len - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.dataset_len / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            gen = np.random.default_rng(self.seed + self.epoch)
            indices = list(self.unshuffled_indices)
            gen.shuffle(indices)
        else:
            indices = list(self.unshuffled_indices)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedWeightedDatasetSampler(Sampler):
    """Weighted random sampler that supports distributed training.

    Uses two-stage sampling (pick dataset, then pick sample within it) to avoid
    storing per-sample weights for large datasets and the torch.multinomial 2^24
    category limit. Works for both distributed and single-GPU (num_replicas=1).
    """
    def __init__(
        self,
        dataset_sizes: list[int],
        dataset_weights: list[float],
        num_samples: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
            assert num_replicas > 0, "World size is not available"
        if rank is None:
            rank = dist.get_rank()
            assert 0 <= rank < num_replicas, "Rank is not available"

        assert len(dataset_sizes) == len(dataset_weights)
        self.dataset_sizes = np.array(dataset_sizes)
        ds_weights = np.array(dataset_weights, dtype=np.float64)
        self.dataset_probs = ds_weights / ds_weights.sum()
        self.dataset_offsets = np.cumsum([0] + list(dataset_sizes[:-1]))
        self.num_samples = num_samples  # total samples across all ranks per epoch
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Pad so num_samples is divisible by num_replicas
        self.total_size = math.ceil(num_samples / num_replicas) * num_replicas
        self.num_samples_per_rank = self.total_size // num_replicas

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        # Stage 1: pick which dataset each sample comes from
        ds_indices = rng.choice(
            len(self.dataset_probs),
            size=self.total_size,
            replace=True,
            p=self.dataset_probs,
        )

        # Stage 2: uniformly pick an index within each chosen dataset
        indices = np.empty(self.total_size, dtype=np.int64)
        for ds_idx in range(len(self.dataset_sizes)):
            mask = ds_indices == ds_idx
            count = int(mask.sum())
            if count > 0:
                indices[mask] = self.dataset_offsets[ds_idx] + rng.integers(
                    0, self.dataset_sizes[ds_idx], size=count
                )

        start = self.rank * self.num_samples_per_rank
        end = start + self.num_samples_per_rank
        rank_indices = indices[start:end]

        return iter(rank_indices.tolist())

    def __len__(self):
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int):
        """Call this at the start of each epoch to get a different shuffle."""
        self.epoch = epoch
