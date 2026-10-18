import numpy as np
from PIL import Image

from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig


class FatalPipelineError(BaseException):
    """
    An error that should crash the rollout pipeline.
    Inherits from BaseException instead of Exception, since the latter
    is universally caught by the pipeline and triggers a retry.
    """
    pass


class PnPPolicyConfig(BasePolicyConfig):
    policy_type: str = "learned"
    action_type: str = "joint_pos"
    policy_cls: type = None

    camera_names: list[str] = ["exo_camera_1", "wrist_camera"]
    action_move_group_names: list[str] = ["arm", "gripper"]
    action_spec: dict[str, int] = {"arm": 7, "gripper": 1}


class PiPnPBenchmarkEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: PnPPolicyConfig = PnPPolicyConfig()
    policy_dt_ms: float = 66.0

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False


def resize_with_crop(images: np.ndarray, height: int, width: int):
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_crop_pil(Image.fromarray(im), height, width, method=Image.BILINEAR) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_crop_pil(image: Image.Image, height: int, width: int, method: int):
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    scale_factor = max(height / cur_height, width / cur_width)
    scaled_h = int(cur_height * scale_factor)
    scaled_w = int(cur_width * scale_factor)
    resized = image.resize((scaled_w, scaled_h), resample=method)

    left = (scaled_w - width) // 2
    top = (scaled_h - height) // 2
    right = left + width
    bottom = top + height
    cropped = resized.crop((left, top, right, bottom))
    assert cropped.size == (width, height)
    return cropped
