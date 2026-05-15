from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from .data import VariantExample, group_examples, split_for_dgao_warm_start
from .metrics import GroupMetrics, compute_group_metrics, normalize_text


def _resolve_device(config: ExperimentConfig) -> torch.device:
    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_aux_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_model_and_tokenizer(config: ExperimentConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    _ensure_pad_token(tokenizer)

    dtype = torch.float16 if torch.cuda.is_available() and config.mixed_precision else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        torch_dtype=dtype,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


def _train_step_loss(
    model,
    tokenizer,
    batch: List[VariantExample],
    device: torch.device,
    max_prompt_length: int,
) -> torch.Tensor:
    prompts = [ex.prompt for ex in batch]
    labels_text = [ex.label for ex in batch]
    targets = [f" {t}" for t in labels_text]

    with tokenizer.as_target_tokenizer() if hasattr(tokenizer, "as_target_tokenizer") else _nullcontext():
        pass

    all_text = [p + t for p, t in zip(prompts, targets)]
    tokenized = tokenizer(
        all_text,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # Mask prompt tokens and optimize only target suffix tokens.
    labels = input_ids.clone()
    for i, p in enumerate(prompts):
        prompt_ids = tokenizer(
            p,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[i, :prompt_len] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


def _iter_batches(records: List[VariantExample], batch_size: int):
    for i in range(0, len(records), batch_size):
        yield records[i : i + batch_size]


def run_sft_training(model, tokenizer, config: ExperimentConfig, train_data: List[VariantExample]) -> Dict[str, float]:
    device = _resolve_device(config)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    effective_bs = max(1, config.global_batch_size)
    grad_accum = max(1, config.grad_accum_steps)

    step = 0
    running_loss = 0.0
    for epoch in range(config.epochs):
        random.shuffle(train_data)
        pbar = tqdm(_iter_batches(train_data, effective_bs), desc=f"SFT epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            loss = _train_step_loss(
                model=model,
                tokenizer=tokenizer,
                batch=batch,
                device=device,
                max_prompt_length=config.max_prompt_length,
            )
            (loss / grad_accum).backward()
            running_loss += float(loss.item())
            step += 1

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % config.log_every_steps == 0:
                pbar.set_postfix(loss=f"{running_loss / max(1, step):.4f}")

    return {"train_loss": running_loss / max(1, step), "train_steps": float(step)}


def _generate_prediction(model, tokenizer, prompt: str, config: ExperimentConfig, device: torch.device) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=config.max_prompt_length)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_new_tokens=config.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0, input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text.split("\n")[0].strip()


def evaluate_model(model, tokenizer, config: ExperimentConfig, eval_data: List[VariantExample]) -> GroupMetrics:
    device = _resolve_device(config)
    model.to(device)

    group_ids: List[int] = []
    preds: List[str] = []
    labels: List[str] = []
    for ex in tqdm(eval_data, desc="Evaluating"):
        pred = _generate_prediction(model, tokenizer, ex.prompt, config, device)
        group_ids.append(ex.group_id)
        preds.append(pred)
        labels.append(ex.label)

    return compute_group_metrics(group_ids=group_ids, preds=preds, labels=labels)


def _logprob_of_response(model, tokenizer, prompt: str, response: str, config: ExperimentConfig, device: torch.device) -> torch.Tensor:
    full = prompt + " " + response
    enc_full = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length,
    )
    enc_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length,
    )

    input_ids = enc_full["input_ids"].to(device)
    attention_mask = enc_full["attention_mask"].to(device)
    prompt_len = min(enc_prompt["input_ids"].shape[1], input_ids.shape[1] - 1)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target = input_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_logp = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(token_logp)
    start = max(prompt_len - 1, 0)
    mask[:, start:] = 1.0
    denom = mask.sum().clamp_min(1.0)
    return (token_logp * mask).sum() / denom


def _reward_from_prediction(task: str, prediction: str, label: str) -> float:
    p = normalize_text(prediction)
    y = normalize_text(label)
    if task == "sst2":
        p = "positive" if "positive" in p else "negative" if "negative" in p else p
        y = "positive" if "positive" in y else "negative" if "negative" in y else y
    return 1.0 if p == y else 0.0


def run_dgao_training(
    model,
    tokenizer,
    config: ExperimentConfig,
    train_data: List[VariantExample],
) -> Dict[str, float]:
    """
    Lightweight DGAO-style training loop for practical reproduction.

    This implementation follows the key paper mechanics for grouped rewards and
    hybrid advantages, while keeping updates simple enough for single-GPU runs.
    """
    device = _resolve_device(config)
    model.to(device)

    warm_data, rl_data = split_for_dgao_warm_start(train_data)

    # Warm start with standard SFT on half of grouped augmented samples.
    warm_stats = run_sft_training(model, tokenizer, config, warm_data)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Reference model for KL regularization.
    ref_device = _resolve_aux_device(config.dgao_reference_device)
    ref_dtype = torch.float16 if ref_device.type == "cuda" and config.mixed_precision else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        cache_dir=config.cache_dir,
        torch_dtype=ref_dtype,
    )
    reference.to(ref_device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    grouped = group_examples(rl_data)
    group_ids = list(grouped.keys())
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    step = 0
    running_loss = 0.0

    for _ in tqdm(range(config.dgao_steps), desc="DGAO"):
        if not group_ids:
            break
        sampled_ids = random.sample(group_ids, k=min(config.dgao_groups_per_step, len(group_ids)))

        samples: List[Tuple[VariantExample, str, float]] = []
        group_reward: Dict[int, float] = {}

        for gid in sampled_ids:
            records = grouped[gid]
            rewards = []
            for ex in records:
                pred = _generate_prediction(model, tokenizer, ex.prompt, config, device)
                r = _reward_from_prediction(ex.task, pred, ex.label)
                samples.append((ex, pred, r))
                rewards.append(r)
            group_reward[gid] = float(sum(rewards) / max(1, len(rewards)))

        baseline = float(sum(group_reward.values()) / max(1, len(group_reward)))

        losses: List[torch.Tensor] = []
        for ex, pred, r in samples:
            a_group = group_reward[ex.group_id] - baseline
            a_ind = r - group_reward[ex.group_id]
            a_hybrid = config.alpha * a_group + (1.0 - config.alpha) * a_ind

            logp_policy = _logprob_of_response(model, tokenizer, ex.prompt, pred, config, device)
            with torch.no_grad():
                logp_ref = _logprob_of_response(reference, tokenizer, ex.prompt, pred, config, ref_device)
                logp_ref = logp_ref.to(logp_policy.device)
            kl_term = (logp_policy - logp_ref)

            # REINFORCE-style objective with KL penalty.
            losses.append(-a_hybrid * logp_policy + config.beta_kl * kl_term)

        if not losses:
            continue

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        step += 1
        running_loss += float(loss.item())

    stats = {
        "warm_start_loss": warm_stats.get("train_loss", 0.0),
        "dgao_loss": running_loss / max(1, step),
        "dgao_steps": float(step),
    }
    return stats


def save_results(
    output_dir: Path,
    config: ExperimentConfig,
    train_stats: Dict[str, float],
    eval_metrics: GroupMetrics,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "train": train_stats,
        "eval": {
            "accuracy": eval_metrics.accuracy,
            "consistency_rate": eval_metrics.consistency_rate,
            "overconfidence_rate": eval_metrics.overconfidence_rate,
        },
    }
    out_file = output_dir / "results.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_file


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
