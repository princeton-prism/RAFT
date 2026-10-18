import openpi.training.config as _config
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer

from src.data.dataset_openpi import MlSpacesDatasetConfigFactory
from src.data.prompt_templates import DEFAULT_PROMPT_TEMPLATES


# Stage II train config (pi0 on joint-position actions, vision tower frozen). The launcher sets
# repo_id, batch size and the number of steps on the CLI.
for _train_cfg in [
    _config.TrainConfig(
        name="molmobot_pi0_droid",
        model=pi0_config.Pi0Config(
            action_dim=8,
            action_horizon=16,
        ),
        data=MlSpacesDatasetConfigFactory(
            # must override repo_id
            joint_pos_actions=True,
            prompt_templates=DEFAULT_PROMPT_TEMPLATES,
            trim_episode_length=10,
        ),
        pytorch_weight_path=None,
        weight_loader=_config.weight_loaders.PaliGemmaWeightLoader(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=300_000,
            decay_lr=5e-5,
        ),
        num_train_steps=200_001,
        batch_size=256,
        log_interval=100,
        save_interval=1000,
        keep_period=10_000,
        num_workers=48,
        freeze_filter="vision_tower",
    ),
]:
    cfg_name = _train_cfg.name
    assert cfg_name not in _config._CONFIGS_DICT, f"Train config {cfg_name} already exists"
    _config._CONFIGS_DICT[cfg_name] = _train_cfg
