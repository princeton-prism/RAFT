import numpy as np
import h5py
import json
from enum import Enum
from dataclasses import dataclass
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import random

import torch
from torch.utils.data import Dataset, default_collate
from torchvision.transforms.v2 import functional as trf_F
from torchvision.transforms import v2 as trf
import decord
from decord import VideoReader

from src.data.prompt_templates import DEFAULT_PROMPT_TEMPLATES


decord.bridge.set_bridge("torch")


class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple


@dataclass
class MlSpacesDatasetMetadata():
    input_features: dict
    output_features: dict
    features: dict
    stats: dict

    def __init__(self):
        self.input_features = {}
        self.output_features = {}
        self.features = {}
        self.stats = {}


def natural_sort(l, key=lambda x: x):
    # taken from https://stackoverflow.com/a/4836734
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=lambda x: alphanum_key(key(x)))


class MlSpacesDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        house_idxs: list[int] | None,
        selected_states: list[str],
        selected_actions: list[str],
        selected_observations: list[str] | None,
        selected_env_states: list[str] | None,
        img_size: tuple[int, int] = (360, 640),
        action_chunking: bool = False,
        obs_horizon: int = 1,
        action_horizon: int = 1,
        drop_n_last_frames: int = 0,
        batch_image_proc: bool = True,
        augment_images: bool = True,
        randomize_prompts: bool = True,
        prompt_templates: dict[str, list[list[str]]] | None = None,
        prompt_sampling_prob_threshold: float = 0.15,
        prompt_sampling_temperature: float = 4.0,
        prompt_sampling_randomize_casing: bool = True,
        prompt_sampling_randomize_punctuation: bool = True,
    ):
        self.data_root = data_root
        self.selected_states = selected_states
        self.selected_actions = selected_actions
        self.selected_observations = selected_observations
        self.selected_env_states = selected_env_states
        self.img_size = img_size
        self.action_chunking = action_chunking
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon if self.action_chunking else 1
        self.drop_n_last_frames = drop_n_last_frames
        self.batch_image_proc = batch_image_proc
        self.augment_images = augment_images
        self.randomize_prompts = randomize_prompts
        self.prompt_templates = (prompt_templates or DEFAULT_PROMPT_TEMPLATES) if randomize_prompts else None
        self.prompt_sampling_prob_threshold = prompt_sampling_prob_threshold
        self.prompt_sampling_temperature = prompt_sampling_temperature
        self.prompt_sampling_randomize_casing = prompt_sampling_randomize_casing
        self.prompt_sampling_randomize_punctuation = prompt_sampling_randomize_punctuation
        self.image_augmentations = trf.Compose([
            trf.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05
            ),
            trf.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            trf.RandomPosterize(bits=7, p=0.2),
            trf.RandomPosterize(bits=6, p=0.2),
            trf.RandomPosterize(bits=5, p=0.2),
            trf.RandomPosterize(bits=4, p=0.2),
            trf.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ])

        # Internal bookkeeping
        self.house_idxs = house_idxs
        self.traj_files: list[str] = []  # [traj_idx] -> file path
        self.traj_file_traj_idxs: list[int] = []  # [traj_idx] -> idx of traj in file
        self.traj_lengths: list[int] = []  # [traj_idx] -> length of traj
        self._build_bookkeeping()
        self.traj_cumsum_lengths = np.cumsum([0] + self.traj_lengths)

        # load fps from data after building bookkeeping
        self.fps = self._get_fps()

        # Metadata
        self.lerobot_meta = MlSpacesDatasetMetadata()
        self._get_features(self[0])
        self.lerobot_meta.stats = self._compute_stats()

    @property
    def selected_action_mgs(self) -> list[str]:
        return [action.split(".")[1] for action in self.selected_actions]

    def _get_fps(self):
        traj_idx = self.traj_file_traj_idxs[0]
        with self._get_file(0) as file:
            obs_scene = json.loads(file[f"traj_{traj_idx}/obs_scene"][()].decode('utf-8').rstrip('\x00'))
            fps: int = round(1000 / obs_scene["policy_dt_ms"])
            assert fps > 0, f"Invalid FPS: {fps}"
            return fps

    def _build_bookkeeping(self):
        with open(self.data_root / "valid_trajectory_index.json", "r") as f:
            valid_traj_index: dict[str, dict[str, list[int]]] = json.load(f)

        n_valid_houses = len(valid_traj_index.keys())
        is_house_dir = lambda d: d.is_dir() and d.name.startswith("house_") and len(list(d.glob("*.h5"))) > 0
        n_saved_houses = sum(1 for house_dir in self.data_root.iterdir() if is_house_dir(house_dir))
        print(f"Using {n_valid_houses}/{n_saved_houses} houses ({n_valid_houses/n_saved_houses:.0%})")
        if n_valid_houses / n_saved_houses < 0.7:
            print("WARN: Less than 70% of houses are marked as valid. Do you need to regenerate the trajectory index?")

        if self.house_idxs is None:
            self.house_idxs = sorted([int(name.split("_")[1]) for name in valid_traj_index.keys()])

        for house_idx in self.house_idxs:
            for datafile_subpath, traj_lens in valid_traj_index[f"house_{house_idx}"].items():
                for traj_key, traj_len in traj_lens.items():
                    traj_idx = int(traj_key.split("_")[1])
                    # subtract 2 to account for the dummy and done actions
                    traj_len = max(0, traj_len - 2 - self.drop_n_last_frames)

                    if traj_len > 0:
                        self.traj_files.append(datafile_subpath)
                        self.traj_file_traj_idxs.append(traj_idx)
                        self.traj_lengths.append(traj_len)

    def _compute_stats(self):
        with open(self.data_root / "aggregated_stats.json", "r") as f:
            aggregated_stats = json.load(f)

        feature_stats = defaultdict(lambda: dict(min=[], max=[], mean=[], std=[]))

        for state in self.selected_states:
            agg_stats_key = "obs/agent/" + state.replace(".", "/")
            feature_stats["observation.state"]["mean"].extend(aggregated_stats[agg_stats_key]["mean"])
            feature_stats["observation.state"]["std"].extend(aggregated_stats[agg_stats_key]["std"])
            feature_stats["observation.state"]["min"].extend(aggregated_stats[agg_stats_key]["min"])
            feature_stats["observation.state"]["max"].extend(aggregated_stats[agg_stats_key]["max"])

        for env_state in (self.selected_env_states or []):
            agg_stats_key = "obs/extra/" + env_state.replace(".", "/")
            feature_stats["observation.environment_state"]["mean"].extend(aggregated_stats[agg_stats_key]["mean"])
            feature_stats["observation.environment_state"]["std"].extend(aggregated_stats[agg_stats_key]["std"])
            feature_stats["observation.environment_state"]["min"].extend(aggregated_stats[agg_stats_key]["min"])
            feature_stats["observation.environment_state"]["max"].extend(aggregated_stats[agg_stats_key]["max"])

        for action in self.selected_actions:
            agg_stats_key = "actions/" + action.replace(".", "/")
            feature_stats["action"]["mean"].extend(aggregated_stats[agg_stats_key]["mean"])
            feature_stats["action"]["std"].extend(aggregated_stats[agg_stats_key]["std"])
            feature_stats["action"]["min"].extend(aggregated_stats[agg_stats_key]["min"])
            feature_stats["action"]["max"].extend(aggregated_stats[agg_stats_key]["max"])

        stats = {}
        for feature_name, feature in self.lerobot_meta.features.items():
            if feature.type != FeatureType.VISUAL:
                stats[feature_name] = feature_stats[feature_name]
            else:
                # use imagenet stats for images
                stats[feature_name] = {
                    "min": np.zeros((3, 1, 1), dtype=np.float32),
                    "max": np.ones((3, 1, 1), dtype=np.float32),
                    "mean": np.array([[0.485, 0.456, 0.406]], dtype=np.float32).reshape(3, 1, 1),
                    "std": np.array([[0.229, 0.224, 0.225]], dtype=np.float32).reshape(3, 1, 1),
                }

        return stats

    def _get_features(self, sample_episode: dict) -> dict[str, PolicyFeature]:
        """Determine LeRobot features from a sample episode."""
        output_features = {}
        input_features = {}

        if self.selected_actions and "action" in sample_episode:
            action_data = sample_episode["action"]
            shape = tuple(action_data.shape[1:])  if self.action_chunking else action_data.shape # Remove time dimension
            output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=shape
            )

        if self.selected_states and "observation.state" in sample_episode:
            state_data = sample_episode["observation.state"]
            shape = tuple(state_data.shape[1:])  # Remove time dimension
            input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=shape
            )

        if (
            self.selected_env_states
            and "observation.environment_state" in sample_episode
        ):
            env_state_data = sample_episode["observation.environment_state"]
            shape = tuple(env_state_data.shape[1:])  # Remove time dimension
            input_features["observation.environment_state"] = PolicyFeature(
                type=FeatureType.ENV,
                shape=shape
            )

        if self.selected_observations:
            for camera_name in self.selected_observations:
                if f"observation.image.{camera_name}" in sample_episode:
                    input_features[f"observation.image.{camera_name}"] = PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, *self.img_size)
                    )

        self.lerobot_meta.input_features = input_features
        self.lerobot_meta.output_features = output_features
        self.lerobot_meta.features = input_features | output_features

    def _get_file(self, traj_idx: int):
        return h5py.File(self.data_root / self.traj_files[traj_idx], "r")

    def _flat_idx_to_traj_idx(self, flat_idx):
        """
        Get the contiguous trajectory index and step index (within that trajectory) from a flat index.
        """
        traj_idx = np.searchsorted(self.traj_cumsum_lengths, flat_idx, side="right") - 1
        # Subtract the end of the previous trajectory
        step = flat_idx - self.traj_cumsum_lengths[traj_idx]

        return traj_idx.item(), step

    def __len__(self):
        return self.traj_cumsum_lengths[-1]

    def _pad_data(self, data: np.ndarray, start_step: int, end_step: int, data_start: int, data_end: int):
        """
        Args:
            data: The actual loaded data (could be subset of trajectory)
            start_step: Desired start index in trajectory coordinates
            end_step: Desired end index in trajectory coordinates (exclusive)
            data_start: Index of the trajectory where data[0] corresponds to
            data_end: Index of the trajectory where data ends (exclusive, so data[-1] is at data_end-1)
        """
        window_size = end_step - start_step
        is_pad = torch.zeros(window_size, dtype=torch.bool)
        
        # Mark which positions in the window need padding
        for i, traj_idx in enumerate(range(start_step, end_step)):
            if traj_idx < data_start or traj_idx >= data_end:
                is_pad[i] = True
        
        # If no padding needed, just return the data
        if not any(is_pad):
            return data, is_pad
        
        # Calculate how much padding we need
        front_pad_length = max(0, data_start - start_step)
        back_pad_length = max(0, end_step - data_end)
        
        # Create padding arrays
        pad_shape_front = (front_pad_length,) + data.shape[1:]
        pad_shape_back = (back_pad_length,) + data.shape[1:]
        
        front_padding = np.zeros(pad_shape_front, dtype=data.dtype)
        back_padding = np.zeros(pad_shape_back, dtype=data.dtype)
        
        # Concatenate: [front_padding, data, back_padding]
        padded_data = np.concatenate([front_padding, data, back_padding], axis=0)
        
        return padded_data, is_pad

    def _get_dict_data(self, keys: list[str], data: h5py.Group, start_step: int, end_step: int):
        traj_start_step = max(0, start_step)
        traj_end_step = None

        all_dict_data: dict[str, list[dict]] = {}
        for key in keys:
            key_data = data[key]
            trajectories = []
            if traj_end_step is None:
                traj_end_step = min(key_data.shape[0], end_step)
            else:
                assert min(key_data.shape[0], end_step) == traj_end_step, f"Mismatch in end step for key {key}"
            for i in range(traj_start_step, traj_end_step):
                json_str = key_data[i].tobytes().decode('utf-8').rstrip('\x00')
                trajectories.append(json.loads(json_str))
            all_dict_data[key] = trajectories

        assert traj_end_step is not None
        return all_dict_data, traj_start_step, traj_end_step

    def _get_state(self, traj_idx, step):
        start_step = step - self.obs_horizon + 1
        end_step = step + 1

        with self._get_file(traj_idx) as file:
            file_traj_idx = self.traj_file_traj_idxs[traj_idx]
            agent_data = file[f"traj_{file_traj_idx}/obs/agent"]
            # Extract unique keys needed for this trajectory
            keys_needed = set()
            for state in self.selected_states:
                keys_needed.add(state.split(".")[0])  # e.g., "qpos" from "qpos.base"
            
            # Load decoded data
            decoded_data, traj_start_step, traj_end_step = self._get_dict_data(list(keys_needed), agent_data, start_step, end_step)
        
        all_state_data = []
        for state in self.selected_states:
            keys = state.split(".")
            top_key = keys[0]  # e.g., "qpos"
            nested_key = keys[1]  # e.g., "base"
            
            state_data = np.array([step_dict[nested_key] for step_dict in decoded_data[top_key]])
            
            # Pad if necessary
            padded_state_data, state_is_pad = self._pad_data(
                state_data, start_step, end_step, traj_start_step, traj_end_step
            )
            all_state_data.append(torch.tensor(padded_state_data, dtype=torch.float32))

        return torch.cat(all_state_data, dim=-1), state_is_pad

    def _get_env_state(self, traj_idx, step):
        start_step = step - self.obs_horizon + 1
        end_step = step + 1

        with self._get_file(traj_idx) as file:
            file_traj_idx = self.traj_file_traj_idxs[traj_idx]
            extra_data = file[f"traj_{file_traj_idx}/obs/extra"]
            # Extract unique keys needed for this trajectory
            keys_needed = set()
            for state in self.selected_env_states:
                keys_needed.add(state.split(".")[0])  # e.g., "door_state" from "door_state.joint_angle"
            
            # Load decoded data
            decoded_data, traj_start_step, traj_end_step = self._get_dict_data(list(keys_needed), extra_data, start_step, end_step)
        
        all_env_state_data = []
        for state in self.selected_env_states:
            keys = state.split(".")
            top_key = keys[0]  # e.g., "door_state_dict"
            nested_key = keys[1]  # e.g., "joint_angle"

            env_state_data = np.array([step_dict[nested_key] for step_dict in decoded_data[top_key]])

            # Pad if necessary
            padded_env_state_data, env_state_is_pad = self._pad_data(
                env_state_data, start_step, end_step, traj_start_step, traj_end_step
            )
            all_env_state_data.append(torch.tensor(padded_env_state_data, dtype=torch.float32)) 

        return torch.cat(all_env_state_data, dim=-1), env_state_is_pad

    def _process_image(self, x: torch.Tensor):
        """
        Resize and center crop the image to the maximum size.
        Args:
            x: The image tensor to process, shape (*, C, H, W).
        Returns:
            The processed image tensor, shape (*, C, H', W').
        """
        target_h, target_w = self.img_size
        h, w = x.shape[-2:]
        if (h, w) != (target_h, target_w):
            scale_factor = max(target_h / h, target_w / w)
            scaled_h = int(h * scale_factor)
            scaled_w = int(w * scale_factor)
            resized = trf_F.resize(x, (scaled_h, scaled_w))
            cropped = trf_F.center_crop(resized, (target_h, target_w))
        else:
            cropped = x

        if self.augment_images:
            cropped = self.image_augmentations(cropped)

        return cropped

    def _get_obs(self, traj_idx: int, step: int, sensor_name: str):
        """Load video frames for observation horizon using decord (much faster than OpenCV)"""
        with self._get_file(traj_idx) as file:
            file_traj_idx = self.traj_file_traj_idxs[traj_idx]
            obs_data = file[f"traj_{file_traj_idx}/obs/sensor_data/{sensor_name}"]
            video_filename = obs_data[:].tobytes().decode('utf-8').rstrip('\x00')
        video_path = str((self.data_root / self.traj_files[traj_idx]).with_name(video_filename))

        # Calculate window
        start_step = step - self.obs_horizon + 1
        end_step = step + 1
        
        # Open video with decord (thread-safe and much faster than OpenCV)
        try:
            vr = VideoReader(video_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open video with decord: {video_path}. Error: {e}")
        
        total_frames = len(vr)
        
        # Determine which frames to actually load
        traj_start_step = max(0, start_step)
        traj_end_step = min(total_frames, end_step)
        assert start_step < total_frames, f"{start_step=} out of bounds for {total_frames=}"
        
        # Load frames using batch reading (MUCH faster than reading one by one)
        if traj_end_step > traj_start_step:
            frame_indices = list(range(traj_start_step, traj_end_step))
            # decord batch_get returns tensor of shape (T, H, W, C) in RGB format
            frames = vr.get_batch(frame_indices).numpy()  # Shape: (num_frames, H, W, 3)
        else:
            # No frames to load, create empty array
            assert len(vr) > 0, f"No frames to load for {traj_idx=}, {step=}, {sensor_name=}"
            height, width = vr[0].shape[:2]
            frames = np.zeros((0, height, width, 3), dtype=np.uint8)

        # Pad if necessary
        padded_frames, obs_is_pad = self._pad_data(
            frames, start_step, end_step, traj_start_step, traj_end_step
        )
        
        # Convert to tensor, keep as uint8 [0, 255]
        # decord already outputs in RGB format, so no color conversion needed
        obs_tensor = torch.from_numpy(padded_frames).permute(0, 3, 1, 2)
        assert obs_tensor.dtype == torch.uint8
        # process image if not batching, otherwise defer to collation
        if not self.batch_image_proc:
            obs_tensor = self._process_image(obs_tensor)

        return obs_tensor, obs_is_pad
            

    def _get_action(self, traj_idx, step):
        start_step = step + 1
        end_step = step + self.action_horizon + 1

        with self._get_file(traj_idx) as file:
            file_traj_idx = self.traj_file_traj_idxs[traj_idx]
            actions_group = file[f"traj_{file_traj_idx}"]["actions"]

            req_act_keys = set()
            for selected_action in self.selected_actions:
                req_act_keys.add(selected_action.split(".")[0])

            data_dict, traj_start_step, traj_end_step = self._get_dict_data(list(req_act_keys), actions_group, start_step, end_step)
            decoded_data: list[dict] = [{} for _ in range(traj_end_step - traj_start_step)]
            for selected_action in self.selected_actions:
                action_key, mg_id = selected_action.split(".")
                assert len(data_dict[action_key]) == len(decoded_data), f"Mismatch in length of data_dict and decoded_data for key {action_key}"
                for i in range(len(data_dict[action_key])):
                    decoded_data[i][mg_id] = data_dict[action_key][i][mg_id]

        all_action_data = []
        for i in range(len(decoded_data)):
            action_data = []
            for move_group in self.selected_action_mgs:
                action_data.append(decoded_data[i][move_group])
            action_data = np.concatenate(action_data)
            all_action_data.append(action_data)

        assert len(all_action_data) > 0
        action_array = np.array(all_action_data, dtype=np.float32)

        padded_action, is_pad = self._pad_data(action_array, start_step, end_step, traj_start_step, traj_end_step)
        # If not doing action chunking, squeeze out the time dimension to get (action_dim,) instead of (1, action_dim)
        if not self.action_chunking and padded_action.shape[0] == 1:
            padded_action = padded_action.squeeze(0)  # (1, action_dim) -> (action_dim,)
            is_pad = is_pad.squeeze(0)
        return padded_action, is_pad

    def _get_task(self, traj_idx):
        with self._get_file(traj_idx) as file:
            file_traj_idx = self.traj_file_traj_idxs[traj_idx]
            obs_scene = json.loads(file[f"traj_{file_traj_idx}/obs_scene"][()].decode('utf-8').rstrip('\x00'))
            if not self.randomize_prompts:
                ret: str = obs_scene["task_description"]
            else:
                task_type: str = obs_scene["task_type"]
                referral_expressions: dict[str, list[tuple[str, float]]] = obs_scene["referral_expressions"]
                sampled_referral_exps: dict[str, str] = {}
                for obj_name in referral_expressions.keys():
                    exps = [exp for exp, prob in referral_expressions[obj_name] if prob > self.prompt_sampling_prob_threshold]
                    # if there are no high-probability expressions, they're all the same-ish so use all of them
                    if len(exps) == 0:
                        exps = [exp for exp, _ in referral_expressions[obj_name]]
                    # if there aren't any expressions, return the default task description
                    if len(exps) == 0:
                        ret: str = obs_scene["task_description"]
                        break
                    # softmax sample with bias towards shorter expressions
                    probs = [np.exp(-len(exp.split()) / self.prompt_sampling_temperature) for exp in exps]
                    probs = np.array(probs) / np.sum(probs)
                    idx = np.random.choice(len(exps), p=probs)
                    sampled_referral_exps[obj_name] = exps[idx]
                else:
                    assert self.prompt_templates is not None
                    prompt_template_group = random.choice(self.prompt_templates[task_type])
                    prompt_template = random.choice(prompt_template_group)
                    prompt = prompt_template.format(**sampled_referral_exps)
                    assert "{" not in prompt and "}" not in prompt, f"Badly formatted prompt: {prompt}"
                    ret = prompt

        if self.prompt_sampling_randomize_casing and random.random() < 0.5:
            ret = ret.lower()
        if self.prompt_sampling_randomize_punctuation and random.random() < 0.5:
            ret = ret.replace(".", "").replace("?", "").replace("!", "")
        return ret

    def __getitem__(self, idx) -> dict:
        sample = {}

        # Metadata
        sample["index"] = idx
        sample["task_index"] = 0  # TODO: set task index
        sample["fetch_start_time"] = time.time()

        traj_idx, step = self._flat_idx_to_traj_idx(idx)
        sample["frame_index"] = step
        sample["episode_index"] = traj_idx
        sample["task"] = self._get_task(traj_idx)
        sample["timestamp"] = step / self.fps

        # Observations
        if self.selected_states:
            sample[f"observation.state"], sample["observation.state_is_pad"] = self._get_state(traj_idx, step)
        if self.selected_env_states:
            sample[f"observation.environment_state"], sample["observation.environment_state_is_pad"] = self._get_env_state(traj_idx, step)
        if self.selected_observations:
            for sensor_name in self.selected_observations:
                sample[f"observation.image.{sensor_name}"], sample[f"observation.image.{sensor_name}_is_pad"] = self._get_obs(traj_idx, step, sensor_name)

        # Actions
        sample["action"], sample["action_is_pad"] = self._get_action(traj_idx, step)

        return sample

    def collate_fn(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = default_collate(samples)
        if self.batch_image_proc:
            for key, value in batch.items():
                if key.startswith("observation.image") and not key.endswith("_is_pad"):
                    batch[key] = self._process_image(value)
        fetch_start_time = batch.pop("fetch_start_time").min().item()
        batch["batch_fetch_time"] = time.time() - fetch_start_time
        return batch

if __name__ == "__main__":
    ds_kwargs = dict(
        selected_states=["qpos.arm", "qpos.gripper"],
        selected_actions=["joint_pos.arm", "joint_pos.gripper"],
        selected_observations=["wrist_camera", "exo_camera_1"],
        img_size=(360, 640),
        drop_n_last_frames=7,
        action_chunking=True,
        obs_horizon=4,
        action_horizon=8,
        selected_env_states=None,
        batch_image_proc=False,
    )

    ds = MlSpacesDataset(
        data_root=Path("data"),
        house_idxs=[0,1,2],
        **ds_kwargs,
    )
    breakpoint()
