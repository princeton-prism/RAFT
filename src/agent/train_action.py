"""
Stage II trainer: PyTorch PI0 training with single- and multi-GPU (DDP) support, using the
openpi train config `molmobot_pi0_droid` (src/model/config_openpi.py) and the MolmoSpaces
data pipeline (src/data/dataset_openpi.py).

Usage (scripts/stage2/train_action.sh wraps this):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> src/agent/train_action.py \
      molmobot_pi0_droid --data.repo-id <action data dir> --pytorch-weight-path <init ckpt dir> \
      --exp-name <run name> --num-train-steps <steps> [--resume]
"""

import contextlib
import dataclasses
import datetime
import gc
import logging
import os
import platform
import shutil
import time
import pickle
import re

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.models.tokenizer import PaligemmaTokenizer

import src.model.config_openpi as _
from src.data.dataset_openpi import create_data_loader


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            tags=["train"],
            job_type="train",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Rank 0 alone does wandb init, the sample-data batch and the val cache, so ranks can
        # reach the first collective far apart; MOLMOBOT_DDP_TIMEOUT_S raises NCCL's default
        # 600 s timeout so that skew does not abort startup.
        timeout_s = int(os.environ.get("MOLMOBOT_DDP_TIMEOUT_S", "3600"))
        torch.distributed.init_process_group(
            backend=backend, init_method="env://", timeout=datetime.timedelta(seconds=timeout_s)
        )

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


@torch.no_grad()
def run_action_validation(model, val_batches, device, seed: int = 1234) -> float:
    """Mean flow-matching loss over a fixed set of held-out (obs, actions)
    batches. The flow noise/timestep RNG is reseeded (and restored) so the
    estimate is comparable across steps without perturbing training."""
    if not val_batches:
        return float("nan")
    was_training = model.training
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    total, n = 0.0, 0
    for observation, actions in val_batches:
        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions = actions.to(dtype=torch.float32, device=device)
        losses = model(observation, actions)
        if isinstance(losses, list | tuple):
            losses = torch.stack(losses)
        elif not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses, device=device, dtype=torch.float32)
        total += float(losses.mean())
        n += 1
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    if was_training:
        model.train()
    return total / max(1, n)


def cleanup_checkpoints(global_step: int, config: _config.TrainConfig, is_main: bool,
                        val_metric_by_step: dict | None = None):
    if not is_main:
        return

    # If KEEP_LAST_CHECKPOINTS=N (>0) and/or KEEP_BEST_CHECKPOINTS=M (>0) are set,
    # retain ONLY the newest N numeric step-checkpoints (latest, for resume) plus
    # the M with the best held-out val loss (from `val_metric_by_step`, lower is
    # better); delete everything else (overrides keep_period). Caps disk use for
    # long resumable runs while preserving the val-optimal checkpoint.
    keep_last = int(os.environ.get("KEEP_LAST_CHECKPOINTS", "0"))
    keep_best = int(os.environ.get("KEEP_BEST_CHECKPOINTS", "0"))
    if keep_best > 0 and keep_last <= 0:
        keep_last = 1  # always keep the latest so resume works
    if keep_last > 0 or keep_best > 0:
        step_dirs = sorted(
            (d for d in config.checkpoint_dir.glob("*") if d.is_dir() and d.name.isdigit()),
            key=lambda d: int(d.name),
        )
        steps = [int(d.name) for d in step_dirs]
        keep = set(steps[-keep_last:]) if keep_last > 0 else set()
        # Explicit steps to pin regardless of recency or val rank (keep_last + keep_best alone
        # would delete a mid-run checkpoint).
        # Accept ':' as well as ',' / whitespace: sbatch --export is itself comma-separated, so a
        # comma inside the value is parsed as the start of the next variable and silently truncates.
        keep |= {int(s) for s in
                 os.environ.get("KEEP_STEPS", "").replace(",", " ").replace(":", " ").split()}
        if keep_best > 0 and val_metric_by_step:
            scored = [(s, val_metric_by_step[s]) for s in steps
                      if s in val_metric_by_step]
            scored.sort(key=lambda x: x[1])  # lower val loss = better
            keep |= {s for s, _ in scored[:keep_best]}
        for ckpt_dir in step_dirs:
            if int(ckpt_dir.name) not in keep:
                logging.info(f"Deleting checkpoint at {ckpt_dir} (keep={sorted(keep)})")
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        return

    # clean up old checkpoints (keeping those specified by keep_period) except the current one
    for i in range(0, global_step, config.save_interval):
        if i % config.keep_period == 0:
            continue

        ckpt_dir = config.checkpoint_dir / f"{i}"
        if ckpt_dir.exists():
            logging.info(f"Deleting checkpoint at {ckpt_dir}")
            shutil.rmtree(ckpt_dir)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: _config.TrainConfig,
    is_main: bool,
    data_config: _config.DataConfig,
    val_metric_by_step: dict | None = None,
):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step. `global_step` has already been
    # incremented by the caller, so the last step of the run is num_train_steps.
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        assets_dir = tmp_ckpt_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(assets_dir / data_config.asset_id, norm_stats)

        # save train config
        with open(assets_dir / "train_config.pkl", "wb") as f:
            pickle.dump(config, f)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        cleanup_checkpoints(global_step, config, is_main, val_metric_by_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def log_sample_data(config: _config.TrainConfig, world_size: int):
    sample_batch_size = min(10 * world_size, config.batch_size)
    sample_config = dataclasses.replace(config, batch_size=sample_batch_size, num_workers=0)
    # Create a separate data loader for sample batch to avoid consuming the main loader
    sample_data_loader = create_data_loader(sample_config, framework="pytorch", shuffle=True)
    observation, _ = next(iter(sample_data_loader))

    paligemma_tokenizer = PaligemmaTokenizer()
    prompts = paligemma_tokenizer._tokenizer.decode(observation.tokenized_prompt.tolist())

    sample_cols = ["prompt"] + sorted(observation.images.keys())
    sample_table = wandb.Table(columns=sample_cols)
    for i in range(len(prompts)):
        row_dict = {
            "prompt": prompts[i],
        }
        for cam_name, images in observation.images.items():
            row_dict[cam_name] = wandb.Image(images[i].permute(1, 2, 0).cpu().numpy())
        row = [row_dict[col] for col in sample_cols]
        sample_table.add_data(*row)
    wandb.log({"sample_data": sample_table}, step=0)


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    if (config.checkpoint_dir / "wandb_id.txt").exists() and not config.resume:
        logging.info(f"Automatically resuming from previous run in {config.checkpoint_dir}")
        config = dataclasses.replace(config, resume=True)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        wandb.summary["checkpoint_dir"] = str(config.checkpoint_dir)

        with open(config.checkpoint_dir / "train_config.pkl", "wb") as f:
            pickle.dump(config, f)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample data to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        log_sample_data(config, world_size)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    original_model = model

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        if is_main:
            logging.info(f"Resumed training from step {global_step}")

    if isinstance(config.freeze_filter, str):
        freeze_regex = re.compile(config.freeze_filter)
        if is_main:
            logging.info(f"Freezing according to filter: {freeze_regex.pattern}")
        n_frozen = 0
        for name, param in model.named_parameters():
            if freeze_regex.search(name):
                param.requires_grad = False
                n_frozen += 1
        if is_main:
            logging.info(f"Froze {n_frozen} parameters")
    elif is_main:
        logging.info("Not freezing any parameters")

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        logging.info("Enabled memory optimizations for 8+ GPU training")

        logging.info("Compiling model with max-autotune")
        model = torch.compile(model, mode="max-autotune", dynamic=False)

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=world_size < 8,  # Not necessary when compiling
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # Gradient accumulation: each optimizer step (== one `global_step`)
    # accumulates `grad_accum` micro-batches. Effective batch size is therefore
    # config.batch_size * grad_accum (config.batch_size is the per-micro-step
    # global batch, split config.batch_size//world_size per GPU by the loader).
    grad_accum = max(1, int(os.environ.get("MOLMOBOT_GRAD_ACCUM", "1")))
    if is_main:
        logging.info(
            f"Gradient accumulation: grad_accum={grad_accum} -> effective_batch_size="
            f"{config.batch_size * grad_accum} (micro global batch={config.batch_size}, "
            f"per-GPU micro={effective_batch_size})"
        )

    model.train()
    start_time = time.time()
    micro_step = 0  # counts micro-batches within the current optimizer step
    accum_loss = 0.0  # sum of unscaled micro-batch losses within the step
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {n_params:,} (trainable: {n_trainable_params:,})")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    assert isinstance(loader, _data_loader.DataLoaderImpl)
    steps_per_epoch = len(loader._data_loader.torch_loader)
    sampler = loader._data_loader.torch_loader.sampler

    # Fixed batches for a comparable val-loss estimate, enabling the latest+2-best
    # retention (KEEP_BEST_CHECKPOINTS). Cached once on CPU; only the main rank validates.
    #
    # VAL_DATA_ROOT (optional): a dataset root holding trajectories EXCLUDED from training.
    # Set it and the loss is a genuine held-out generalization estimate, so a rising curve
    # means overfitting and a still-falling one means the step budget is too small. Leave it
    # unset and the batches come from the training loader -- comparable across steps (which
    # is all best-checkpoint selection needs) but blind to overfitting, since a memorizing
    # model drives that number down too.
    # The val root must carry the TRAIN split's aggregated_stats.json so both use one
    # normalizer; see scripts/data/build_open_holdout_split.py.
    val_batches = []
    val_metric_by_step: dict[int, float] = {}
    val_n = int(os.environ.get("VAL_BATCHES", "0"))
    val_root = os.environ.get("VAL_DATA_ROOT", "").strip()
    val_interval = int(os.environ.get("VAL_INTERVAL", "0")) or config.save_interval
    if is_main and val_n > 0:
        if val_root:
            import dataclasses as _dc

            vcfg = _dc.replace(config, data=_dc.replace(config.data, repo_id=val_root))
            vloader, _ = build_datasets(vcfg)
            vi = iter(vloader)
            logging.info(f"[val] HELD-OUT root: {val_root}")
        else:
            vi = iter(loader)
            logging.info("[val] no VAL_DATA_ROOT -- batches drawn from the TRAIN pool "
                         "(comparable across steps, but NOT a generalization estimate)")
        for _ in range(val_n):
            try:
                vobs, vact = next(vi)
            except StopIteration:
                break
            vobs = jax.tree.map(lambda x: x.detach().to("cpu"), vobs)
            val_batches.append((vobs, vact.detach().to("cpu")))
        del vi
        logging.info(f"[val] cached {len(val_batches)} batches "
                     f"({'held-out' if val_root else 'train-pool'})")

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(global_step // steps_per_epoch)

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break
            if hasattr(sampler, "set_epoch"):
                # add one so we can set the epoch before we reconstruct the iterator next step
                sampler.set_epoch((global_step + 1) // steps_per_epoch)

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR (schedule is on optimizer steps == global_step)
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Whether this micro-batch closes an accumulation window.
            is_boundary = (micro_step + 1) % grad_accum == 0

            # Forward + backward. On non-boundary micro-steps under DDP, skip the
            # gradient all-reduce (no_sync) so we only sync once per optimizer
            # step. Loss is scaled by 1/grad_accum so the accumulated gradient
            # equals the mean over the full effective batch.
            sync_ctx = model.no_sync() if (use_ddp and not is_boundary) else contextlib.nullcontext()
            with sync_ctx:
                losses = model(observation, actions)
                if isinstance(losses, list | tuple):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(losses, device=device, dtype=torch.float32)
                loss = losses.mean()
                (loss / grad_accum).backward()

            accum_loss += loss.item()
            micro_step += 1

            # Log memory usage after backward pass (first few optimizer steps)
            if global_step < 5 and is_main and torch.cuda.is_available() and is_boundary:
                log_memory_usage(device, global_step, "after_backward")

            # Keep accumulating until the window closes.
            if not is_boundary:
                continue

            # ---- accumulation boundary: take the optimizer step ----
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            step_loss = accum_loss / grad_accum
            accum_loss = 0.0
            micro_step = 0

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": step_loss,
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                        "epoch": global_step / steps_per_epoch,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step, commit=True)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Held-out val loss at save boundaries -> ranks the 2-best checkpoints. VAL_INTERVAL
            # lets validation run more often than checkpoints are written.
            if val_batches and is_main and (
                (global_step % val_interval == 0 and global_step > 0)
                or global_step == config.num_train_steps
            ):
                vloss = run_action_validation(original_model, val_batches, device)
                val_metric_by_step[global_step] = vloss
                logging.info(f"[val] step={global_step} val_action_loss={vloss:.4f}")
            # Save checkpoint using the new mechanism
            save_checkpoint(original_model, optim, global_step, config, is_main,
                            data_config, val_metric_by_step)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{step_loss:.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    # openpi's TrainConfig.freeze_filter is a tyro.conf.Suppress[Filter], so it
    # cannot be set on the CLI. Allow overriding it with a string regex via the
    # FREEZE_FILTER env var; the freezing logic in train_loop handles the str case.
    freeze_filter_env = os.environ.get("FREEZE_FILTER")
    if freeze_filter_env:
        config = dataclasses.replace(config, freeze_filter=freeze_filter_env)
    train_loop(config)


if __name__ == "__main__":
    main()
