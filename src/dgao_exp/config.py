from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dataset_name: str = "sst2"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    local_dataset_path: Optional[str] = None
    train_split_name: str = "train"
    eval_split_name: str = "validation"

    # Data construction
    train_groups: int = 200
    eval_groups: int = 100
    num_order_variants: int = 8
    math_shots: int = 4
    sentiment_shots: int = 16
    rag_docs: int = 5
    seed: int = 42
    max_prompt_length: int = 1024
    max_new_tokens: int = 64

    # Training
    mode: str = "sft"  # sft | paft | dgao
    lr: float = 2e-5
    epochs: int = 1
    global_batch_size: int = 8
    grad_accum_steps: int = 1
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # DGAO-specific
    alpha: float = 0.5
    beta_kl: float = 0.05
    clip_eps: float = 0.2
    eps_adv: float = 1e-8
    temperature: float = 1.0
    top_p: float = 1.0
    dgao_steps: int = 200
    dgao_groups_per_step: int = 4
    dgao_reference_device: str = "cpu"  # cpu | cuda | auto

    # Runtime
    device: str = "auto"  # auto | cuda | cpu
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    log_every_steps: int = 10
    save_every_steps: int = 100


    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir)
