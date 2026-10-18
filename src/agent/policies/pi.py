import os
from pathlib import Path
from dataclasses import dataclass
import pickle
import time
from typing import cast
import logging
import dataclasses

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.shared import download
from openpi.models import pi0_config

from molmo_spaces.policy.base_policy import StatefulPolicy, InferencePolicy
from molmo_spaces.policy.learned_policy.utils import resize_with_pad

from src.agent.eval_utils import resize_with_crop
import src.model.config_openpi as _


log = logging.getLogger(__name__)

# The released checkpoints pickled their TrainConfig while this code still lived in a
# `molmobot_pi0` package; map those module paths to their new homes so
# assets/train_config.pkl keeps loading.
_LEGACY_MODULES = {
    "molmobot_pi0.dataset_openpi": "src.data.dataset_openpi",
    "molmobot_pi0.dataset": "src.data.dataset",
    "molmobot_pi0.prompt_templates": "src.data.prompt_templates",
    "molmobot_pi0.config_openpi": "src.model.config_openpi",
}


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(_LEGACY_MODULES.get(module, module), name)


EXO_CAM_INPUT_KEY = "observation/exterior_image_1_left"
WRIST_CAM_INPUT_KEY = "observation/wrist_image_left"


@dataclass
class PiPolicyState:
    actions_buffer: np.ndarray | None
    current_buffer_index: int
    last_prompt: str


class PiPolicy(InferencePolicy, StatefulPolicy):
    def __init__(
        self,
        model_name: str | None = None,
        checkpoint_dir: str | None = None,
        use_torch: bool = True,
        buffer_length: int | None = None,
        cameras: dict[str, str] | None = None,
        device_id: int | None = None,
        compile_mode: str | None = None,
        max_joint_delta: float | list[float] | None = None,
    ) -> None:
        super().__init__(None, "pick_and_place")

        self.buffer_length = buffer_length if buffer_length and buffer_length > 0 else 1000
        self._prepared = False
        self._last_prompt: str | None = None
        self._use_torch = use_torch
        self._device_id = device_id
        if max_joint_delta is None or isinstance(max_joint_delta, float):
            self.max_joint_delta = max_joint_delta
        else:
            self.max_joint_delta = np.array(max_joint_delta)

        if not checkpoint_dir:
            assert model_name is not None, "model_name is required when checkpoint_dir is not provided"
            self._default_config = True
            self.pi_config = _config.get_config(model_name)
            if use_torch:
                log.info(f"Loading PyTorch weights from {self.pi_config.pytorch_weight_path}")
                checkpoint_dir = self.pi_config.pytorch_weight_path
            elif isinstance(self.pi_config.weight_loader, _config.weight_loaders.CheckpointWeightLoader):
                ckpt_path = os.path.dirname(self.pi_config.weight_loader.params_path)
                log.info(f"Loading jax weights from {ckpt_path}")
                checkpoint_dir = str(download.maybe_download(ckpt_path))
            else:
                weight_url = f"gs://openpi-assets/checkpoints/{model_name}"
                checkpoint_dir = str(download.maybe_download(weight_url))
        else:
            self._default_config = False
            config_file = Path(checkpoint_dir) / "assets/train_config.pkl"
            if config_file.exists():
                with open(config_file, "rb") as f:
                    self.pi_config = _LegacyUnpickler(f).load()
                assert isinstance(self.pi_config, _config.TrainConfig)
                log.info(f"Loaded train config from {config_file}")
                if model_name is not None:
                    assert self.pi_config.name == model_name, f"Model name mismatch: {self.pi_config.name} != {model_name}"
                else:
                    model_name = self.pi_config.name
            else:
                raise ValueError(f"No train config found for checkpoint {checkpoint_dir}")

        self.model_name = model_name

        assert isinstance(self.pi_config.model, pi0_config.Pi0Config)
        self.pi_config = dataclasses.replace(
            self.pi_config,
            model=dataclasses.replace(
                self.pi_config.model,
                pytorch_compile_mode=compile_mode,
            ),
        )

        if cameras is not None:
            self._cameras = cameras
        else:
            log.warning("No cameras specified and could not infer from config, using default cameras")
            self._cameras = {
                "exo_camera_1": EXO_CAM_INPUT_KEY,
                "wrist_camera": WRIST_CAM_INPUT_KEY,
            }

        assert checkpoint_dir and os.path.exists(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} does not exist"
        self.checkpoint_dir = Path(checkpoint_dir)

        self.reset()

    def reset(self, *args, **kwargs):
        self.actions_buffer: np.ndarray | None = None
        self.current_buffer_index: int = 0
        self._last_prompt = None

    def get_state(self):
        state = PiPolicyState(
            actions_buffer=self.actions_buffer.copy() if self.actions_buffer is not None else None,
            current_buffer_index=self.current_buffer_index,
            last_prompt=self._last_prompt,
        )
        return state

    def set_state(self, state: PiPolicyState):
        self.actions_buffer = state.actions_buffer.copy() if state.actions_buffer is not None else None
        self.current_buffer_index = state.current_buffer_index
        self._last_prompt = state.last_prompt

    def prepare_model(self):
        if self._prepared:
            return
        self.model = _policy_config.create_trained_policy(
            self.pi_config,
            self.checkpoint_dir,
            pytorch_device=f"cuda:{self._device_id}" if self._device_id is not None else None,
        )
        self.reset()
        self._prepared = True

        log.info("Warming up model...")
        dummy_input = {
            EXO_CAM_INPUT_KEY: np.zeros((224, 224, 3), dtype=np.uint8),
            WRIST_CAM_INPUT_KEY: np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros((7,), dtype=np.float32),
            "observation/gripper_position": np.zeros((1,), dtype=np.float32),
            "prompt": "place the red block on the green block",
        }
        self.model.infer(dummy_input)
        log.info("Model warmup complete.")

    def has_queued_actions(self) -> bool:
        return self.actions_buffer is not None and self.current_buffer_index < min(self.buffer_length, len(self.actions_buffer))

    def obs_to_model_input(self, obs):
        if isinstance(obs, list):
            # TODO: obs shouldn't be a list, this is a bug in MolmoSpaces
            obs = obs[0]
        if "task" in obs:
            prompt = obs["task"]
            assert self._last_prompt is None or self._last_prompt == prompt, f"Prompt changed during inference! {self._last_prompt} -> {prompt}"
        elif self._last_prompt is not None:
            prompt = self._last_prompt
        elif self.task is not None:
            prompt = self.task.get_task_description()
        else:
            raise ValueError("No prompt passed and no task registered!")

        if self._last_prompt is None:
            self._last_prompt = prompt

        # skip image processing if we're not going to run inference
        images = {}
        if not self.has_queued_actions():
            for cam_obs_key, cam_input_key in self._cameras.items():
                # resize to 360p to match data
                img = resize_with_crop(obs[cam_obs_key], 360, 640)
                images[cam_input_key] = resize_with_pad(img, 224, 224)

        arm = np.array(obs["qpos"]["arm"])
        gripper = np.array(obs["qpos"]["gripper"])
        model_input = {
            "observation/joint_position": arm,
            "observation/gripper_position": gripper,
            "prompt": prompt,
            **images
        }
        return model_input

    def model_output_to_action(self, model_output: np.ndarray) -> dict[str, np.ndarray]:
        gripper_pos = np.clip(model_output[7], 0.0, 1.0) * np.array([255.0])
        arm_output = model_output[:7].reshape(7)
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "openpi"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.buffer_length
        info["prompt"] = self._last_prompt
        info["timestamp"] = time.time()
        return info


class PiJointVelPolicy(PiPolicy):
    def inference_model(self, model_input):
        self.prepare_model()
        if not self.has_queued_actions():
            self.actions_buffer = cast(np.ndarray, self.model.infer(model_input)["actions"])
            self.current_buffer_index = 0

            max_vels = np.max(np.abs(self.actions_buffer), axis=-1)
            scales = np.maximum(max_vels, np.ones_like(max_vels))
            self.actions_buffer[:, :7] /= scales.reshape(-1, 1)

        assert self.actions_buffer is not None
        model_output = self.actions_buffer[self.current_buffer_index]
        max_delta = 0.2 if self.max_joint_delta is None else self.max_joint_delta
        model_output[:7] = model_input["observation/joint_position"] + model_output[:7] * max_delta
        self.current_buffer_index += 1
        return model_output


class PiJointPosPolicy(PiPolicy):
    def inference_model(self, model_input):
        self.prepare_model()
        if not self.has_queued_actions():
            self.actions_buffer = cast(np.ndarray, self.model.infer(model_input)["actions"])
            self.current_buffer_index = 0

        assert self.actions_buffer is not None
        model_output = self.actions_buffer[self.current_buffer_index]

        if self.max_joint_delta is not None:
            predicted_deltas = model_output[:7] - model_input["observation/joint_position"]
            relative_scale = np.abs(predicted_deltas) / self.max_joint_delta

            if np.max(relative_scale) > 1:
                predicted_deltas /= np.max(relative_scale)
                model_output[:7] = model_input["observation/joint_position"] + predicted_deltas
        
        self.current_buffer_index += 1
        return model_output
