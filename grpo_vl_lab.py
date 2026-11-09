"""
GRPO for VLMs on ChartQA, using Qwen3-VL-2B-Instruct as the policy model.
"""

import argparse
import contextlib
import csv
import functools
import math
import os
import time

import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoProcessor

try:  # transformers 4.x
    from transformers import AutoModelForImageTextToText as AutoVLM
except ImportError:  # newer releases
    from transformers import AutoModelForMultimodalLM as AutoVLM

INSTRUCTION = "Answer the question using a single word or phrase."


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["ddp", "fsdp"], default="fsdp")
    p.add_argument(
        "--model", default="Qwen/Qwen3-VL-2B-Instruct", help="Hub name or local folder"
    )
    p.add_argument(
        "--data", default="HuggingFaceM4/ChartQA", help="Hub name or local folder"
    )
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument(
        "--image-size", type=int, default=640, help="longest image side in pixels"
    )
    p.add_argument("--epochs", type=int, default=1)
    # GRPO
    p.add_argument(
        "--num-generations", type=int, default=8, help="G: answers sampled per question"
    )
    p.add_argument(
        "--prompts-per-gpu", type=int, default=2, help="questions per GPU per step"
    )
    p.add_argument(
        "--micro-batch", type=int, default=4, help="sequences per forward/backward"
    )
    p.add_argument(
        "--temperature", type=float, default=1.0, help="sampling temperature"
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=16, help="for rollouts and eval"
    )
    p.add_argument(
        "--kl-beta",
        type=float,
        default=0.0,
        help="KL penalty to the starting model; 0 = none",
    )
    p.add_argument("--clip-eps", type=float, default=0.2, help="PPO ratio clipping")
    p.add_argument(
        "--ppo-epochs", type=int, default=1, help="optimizer steps per rollout batch"
    )
    p.add_argument(
        "--no-std-norm",
        action="store_true",
        help="A = r - mean, without / std (Dr. GRPO)",
    )
    # Optimization
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--max-steps", type=int, default=0, help="stop early (0 = run all epochs)"
    )
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument(
        "--ac", action="store_true", help="activation checkpointing on every block"
    )
    p.add_argument(
        "--train-vision", action="store_true", help="also train the vision encoder"
    )
    p.add_argument(
        "--skip-baseline", action="store_true", help="skip the eval before training"
    )
    p.add_argument(
        "--save-dir", default=None, help="save the trained model in Hugging Face format"
    )
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--exp-name", type=str, default="grpo_lab", help="expreiment name")
    return p.parse_args()


# ----------------------------------------------------------------------------- data


def answer_of(example):
    label = example["label"]
    return str(label[0] if isinstance(label, (list, tuple)) else label).strip()


def prep_image(image, max_side):
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))  # keeps aspect ratio, only ever shrinks
    return image


def user_turn(question):
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{question}\n{INSTRUCTION}"},
        ],
    }


def collate_prompts(examples, processor, max_side, repeat=1):
    """Left-padded prompts ending with the assistant header; each example repeated `repeat` times
    in a row, so rows [k*G, (k+1)*G) are the G samples of question k."""
    examples = [ex for ex in examples for _ in range(repeat)]
    processor.tokenizer.padding_side = (
        "left"  # generate() appends new tokens on the right
    )
    texts = [
        processor.apply_chat_template(
            [user_turn(ex["query"])], tokenize=False, add_generation_prompt=True
        )
        for ex in examples
    ]
    images = [prep_image(ex["image"], max_side) for ex in examples]
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return batch, [answer_of(ex) for ex in examples], [ex["query"] for ex in examples]


def slice_rows(batch, start, end):
    """Rows [start, end) of a processor batch with one image per row. pixel_values holds the
    patches of all images back to back (t*h*w rows per image), so it is cut by patch counts."""
    patches = batch["image_grid_thw"].prod(-1).cumsum(0).tolist()
    p0 = patches[start - 1] if start else 0
    out = {}
    for k, v in batch.items():
        out[k] = v[p0 : patches[end - 1]] if k == "pixel_values" else v[start:end]
    return out


# ----------------------------------------------------------------------------- metric


def to_float(text):
    try:
        return float(text.rstrip("%")) / 100 if text.endswith("%") else float(text)
    except ValueError:
        return None


def relaxed_correct(pred, target, tol=0.05):
    """ChartQA relaxed accuracy: numbers within 5% count as correct, text must match exactly."""
    pred = pred.strip().rstrip(".").strip()
    p, t = to_float(pred.replace(",", "")), to_float(target.replace(",", ""))
    if (
        p is not None and t
    ):  # a target of 0 falls back to exact match, as in the reference code
        return abs(p - t) / abs(t) <= tol
    return pred.lower() == target.lower()


@contextlib.contextmanager
def root_gathered(model):
    """Gather the root FSDP unit (embeddings, patch embedding, mergers, lm_head) for generate().
    transformers >= 5 runs the vision encoder in generate() before calling model.forward(), so
    FSDP's root pre-forward hook, which normally initializes FSDP and gathers these weights,
    has not run yet: the patch embedding would see sharded weights and fail. Harmless on 4.x."""
    if not isinstance(model, FSDPModule):
        yield
        return
    model._get_fsdp_state()._lazy_init()  # make the model the FSDP root before any block runs
    model.unshard()
    try:
        yield
    finally:
        model.reshard()


@torch.no_grad()
def evaluate(model, gen_model, loader, processor, device, synced, max_new_tokens, rank):
    model.eval()
    correct = total = 0
    shown = 0
    for batch, targets, questions in loader:
        batch = batch.to(device)
        # synced_gpus: under FSDP every forward is a collective, so all ranks must keep
        # stepping until the slowest rank has finished, or the all-gathers deadlock.
        with root_gathered(model):
            out = gen_model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                synced_gpus=synced,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        preds = processor.batch_decode(
            out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for q, p, t in zip(questions, preds, targets):
            ok = relaxed_correct(p, t)
            correct += ok
            total += 1
            if rank == 0 and shown < 3:
                print(
                    f"  Q: {q}\n  target: {t!r}  pred: {p.strip()!r}  {'OK' if ok else 'X'}",
                    flush=True,
                )
                shown += 1
    counts = torch.tensor([correct, total], dtype=torch.float64, device=device)
    dist.all_reduce(counts)
    model.train()
    return counts[0].item() / counts[1].item()


# ----------------------------------------------------------------------------- GRPO pieces


@torch.no_grad()
def rollout(gen_model, batch, processor, args, synced, eos_ids, vision_ids):
    """Sample one answer per row. Returns the answer tokens [B, L] and their mask [B, L]
    (1 up to and including the first end-of-turn token, 0 for the padding after it)."""
    with root_gathered(gen_model):
        out = gen_model.generate(
            **batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=1.0,  # sample from the model's real distribution: the loss scores exactly
            top_k=0,  # these probabilities, so no top-k / top-p truncation
            repetition_penalty=1.0,
            synced_gpus=synced,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    completion = out[:, batch["input_ids"].shape[1] :]
    L = completion.shape[1]
    is_eos = torch.isin(completion, eos_ids)
    first_eos = torch.where(is_eos.any(1), is_eos.int().argmax(1), L)
    pos = torch.arange(L, device=completion.device)[None]
    mask = (pos <= first_eos[:, None]).long()
    # Sampling from the full vocabulary can (rarely) produce a vision token such as <|image_pad|>.
    # In the training forward, Qwen would count it as an extra image slot and fail
    # ("Image features and image tokens do not match"). Cut the answer before it.
    is_vision = torch.isin(completion, vision_ids)
    if is_vision.any():
        first_vision = torch.where(is_vision.any(1), is_vision.int().argmax(1), L)
        mask = mask * (pos < first_vision[:, None]).long()
        completion = completion.masked_fill(is_vision, processor.tokenizer.pad_token_id)
    return completion, mask


def with_completion(batch, completion, mask):
    """Prompt batch + sampled answers -> one batch for a single forward over prompt and answer."""
    prompt_shape = batch["input_ids"].shape
    full = {}
    for k, v in batch.items():
        if k == "input_ids":
            full[k] = torch.cat([v, completion], 1)
        elif k == "attention_mask":
            full[k] = torch.cat([v, mask.to(v.dtype)], 1)
        elif (
            torch.is_tensor(v) and v.shape == prompt_shape
        ):  # other per-token ids, e.g. mm_token_type_ids
            full[k] = torch.cat([v, torch.zeros_like(completion, dtype=v.dtype)], 1)
        else:  # pixel_values, image_grid_thw
            full[k] = v
    return full


def answer_logps(model, mb, L, temperature):
    """Log-prob of each answer token under `model`, shape [b, L].
    The logit at position t predicts token t+1, so the L answer tokens are predicted by the
    last L+1 positions minus the very last one. logits_to_keep computes only those: [b, L+1, V]
    instead of [b, prompt+L, V], which is what keeps memory low."""
    logits = model(**mb, use_cache=False, logits_to_keep=L + 1).logits[:, :-1].float()
    logits = (
        logits / temperature
    )  # score the distribution the answers were sampled from
    tokens = mb["input_ids"][:, -L:]
    return logits.gather(-1, tokens[..., None]).squeeze(-1) - logits.logsumexp(-1)


def group_advantages(rewards, G, std_norm):
    r = rewards.view(-1, G)
    adv = r - r.mean(1, keepdim=True)
    if std_norm:
        adv = adv / (r.std(1, keepdim=True) + 1e-4)
    no_signal = (r.std(1) == 0).sum()  # groups where every answer got the same reward
    return adv.view(-1), no_signal


def shard_model(model, mode, device, local_rank, use_cuda, block_names, ac):
    if ac:
        # Wrap each block in a checkpoint wrapper BEFORE sharding, so FSDP's hooks sit outside it.
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda m: type(m).__name__ in block_names,
        )
        blocks = [m for m in model.modules() if isinstance(m, CheckpointWrapper)]
    else:
        blocks = [m for m in model.modules() if type(m).__name__ in block_names]
    if not blocks:
        raise RuntimeError(f"No transformer blocks found; looked for {block_names}")
    if mode == "fsdp":
        for block in blocks:
            fully_shard(block)
        fully_shard(model)
        return model, model, blocks
    model.to(device)
    if not any(p.requires_grad for p in model.parameters()):  # frozen reference model
        return model, model, blocks
    ddp = DDP(model, device_ids=[local_rank] if use_cuda else None)
    return (
        ddp,
        ddp.module,
        blocks,
    )  # generate() on the plain module; DDP only wraps forward


# ----------------------------------------------------------------------------- main


def main():
    args = parse_args()
    G = args.num_generations
    if G < 2:
        raise ValueError(
            "--num-generations must be at least 2: advantages compare answers within a group"
        )

    use_cuda = torch.cuda.is_available()  # CPU + gloo works too, for debugging only
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    def barrier():
        dist.barrier(device_ids=[local_rank]) if use_cuda else dist.barrier()

    def sync():
        if use_cuda:
            torch.cuda.synchronize(device)

    if rank != 0:
        transformers.logging.set_verbosity_error()
    torch.manual_seed(args.seed)

    # 1. Data. Rank 0 loads first, so only one process downloads and caches.
    if rank != 0:
        barrier()
    train_ds = load_dataset(args.data, split="train").shuffle(seed=args.seed)
    train_ds = train_ds.select(range(min(args.train_samples, len(train_ds))))
    eval_ds = load_dataset(args.data, split="test").shuffle(seed=args.seed)
    eval_ds = eval_ds.select(range(min(args.eval_samples, len(eval_ds))))
    if rank == 0:
        barrier()

    processor = AutoProcessor.from_pretrained(args.model)

    # 2. Policy model (trained), and optionally a frozen reference copy for the KL penalty.
    #    fp32 on purpose: TITAN RTX (Turing) has no bf16, and fp16 needs loss scaling.
    model = AutoVLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    )
    if not args.train_vision:
        for name, param in model.named_parameters():
            if "visual" in name and "merger" not in name:
                param.requires_grad_(False)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    block_names = set(getattr(model, "_no_split_modules", None) or [])

    # Tokens that end an answer: <|im_end|> for Qwen chat models, plus the tokenizer's EOS.
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {
        processor.tokenizer.eos_token_id
    }
    eos_ids = torch.tensor(sorted(i for i in eos if i is not None), device=device)
    vision_ids = [
        getattr(model.config, k, None)
        for k in [
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]
    ]
    vision_ids = torch.tensor([i for i in vision_ids if i is not None], device=device)

    ref_model = None
    if args.kl_beta > 0:
        ref_model = AutoVLM.from_pretrained(
            args.model, dtype=torch.float32, attn_implementation="sdpa"
        )
        ref_model.requires_grad_(False)
        ref_model, _, _ = shard_model(
            ref_model, args.mode, device, local_rank, use_cuda, block_names, ac=False
        )
        ref_model.eval()

    # 3. Distribute the policy.
    model, gen_model, blocks = shard_model(
        model, args.mode, device, local_rank, use_cuda, block_names, args.ac
    )
    if rank == 0:
        print(
            f"[{args.mode}] {n_total / 1e9:.2f}B params, {n_train / 1e9:.2f}B trainable; {len(blocks)} blocks; "
            f"reference model: {'yes, beta=' + str(args.kl_beta) if ref_model is not None else 'no'}",
            flush=True,
        )

    # 4. Data loaders. Every rank gets the same number of questions per step (drop_last), so
    #    every rank runs the same number of generate() calls and forward/backward passes.
    sampler = DistributedSampler(
        train_ds,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.prompts_per_gpu,
        sampler=sampler,
        drop_last=True,
        num_workers=2,
        collate_fn=functools.partial(
            collate_prompts, processor=processor, max_side=args.image_size, repeat=G
        ),
    )
    n_eval = len(eval_ds) // world * world  # equal share per rank
    if n_eval == 0:
        raise ValueError("--eval-samples must be at least the number of GPUs")
    eval_loader = DataLoader(
        Subset(eval_ds, list(range(rank, n_eval, world))),
        batch_size=args.eval_batch_size,
        num_workers=2,
        collate_fn=functools.partial(
            collate_prompts, processor=processor, max_side=args.image_size
        ),
    )
    synced = args.mode == "fsdp" and world > 1

    # 5. Optimizer on trainable params, created AFTER sharding (so it holds the shards).
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = (
        len(train_loader) * args.epochs
    )  # rollout steps; each makes --ppo-epochs optimizer steps
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if total_steps == 0:
        raise ValueError("Not enough training samples for GPUs x prompts per GPU")
    warmup = max(1, int(args.warmup_ratio * total_steps * args.ppo_epochs))
    total_updates = total_steps * args.ppo_epochs

    def lr_lambda(step):  # linear warmup, then cosine decay to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, total_updates - warmup))
        )

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # 6. Baseline accuracy (greedy decoding, like the final eval).
    acc_before = None
    if not args.skip_baseline:
        t0 = time.perf_counter()
        acc_before = evaluate(
            model,
            gen_model,
            eval_loader,
            processor,
            device,
            synced,
            args.max_new_tokens,
            rank,
        )
        if rank == 0:
            print(
                f"Accuracy before training: {acc_before:.1%} on {n_eval} test questions "
                f"({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
        if use_cuda:
            torch.cuda.empty_cache()  # drop eval's odd-sized cached blocks (less fragmentation)

    # 7. Train.
    rows = args.prompts_per_gpu * G
    chunks = [
        (s, min(s + args.micro_batch, rows)) for s in range(0, rows, args.micro_batch)
    ]
    if rank == 0:
        print(
            f"Training {total_steps} steps; per step: {args.prompts_per_gpu} questions x {G} answers x "
            f"{world} GPUs = {rows * world} answers, {len(chunks)} micro-batches of <= {args.micro_batch} "
            f"per GPU, {args.ppo_epochs} optimizer step(s)",
            flush=True,
        )
        print(
            f"{'step':>9}  {'reward':>6}  {'no-signal':>9}  {'len':>4}  {'kl':>6}  {'clip':>5}  "
            f"{'gnorm':>6}  {'lr':>8}  rollout/train s",
            flush=True,
        )
    torch.manual_seed(args.seed + rank)  # different sampling noise on each rank
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    # Running sums over the log window: reward, answers, no-signal groups, groups,
    # answer tokens, KL * tokens, clipped tokens, tokens seen by the clip check
    stats = torch.zeros(8, dtype=torch.float64, device=device)
    t_roll = t_train = 0.0
    step, answer_tokens = 0, 0
    sync()
    t_start = time.perf_counter()
    done = False
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch, targets, questions in train_loader:
            batch = batch.to(device)

            # --- 1-3. Rollout, reward, advantage
            t0 = time.perf_counter()
            model.eval()
            completion, mask = rollout(
                gen_model, batch, processor, args, synced, eos_ids, vision_ids
            )
            model.train()
            preds = processor.batch_decode(completion, skip_special_tokens=True)
            rewards = torch.tensor(
                [float(relaxed_correct(p, t)) for p, t in zip(preds, targets)],
                device=device,
            )
            adv, no_signal = group_advantages(rewards, G, not args.no_std_norm)
            full = with_completion(batch, completion, mask)
            L = completion.shape[1]
            n_tok = mask.sum().clamp(
                min=1
            )  # loss = sum over answer tokens / n_tok, whole rollout
            answer_tokens += int(n_tok)

            ref_logps = [None] * len(chunks)
            if ref_model is not None:
                with torch.no_grad():
                    for j, (s, e) in enumerate(chunks):
                        ref_logps[j] = answer_logps(
                            ref_model, slice_rows(full, s, e), L, args.temperature
                        )
            sync()
            t1 = time.perf_counter()
            t_roll += t1 - t0

            # --- 4. Policy update
            old_logps = [None] * len(chunks)
            for it in range(args.ppo_epochs):
                for j, (s, e) in enumerate(chunks):
                    logp = answer_logps(
                        model, slice_rows(full, s, e), L, args.temperature
                    )
                    if it == 0:  # parameters are still those that sampled the rollout
                        old_logps[j] = logp.detach()
                    m, a = mask[s:e].float(), adv[s:e, None]
                    ratio = torch.exp(logp - old_logps[j])
                    clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                    per_tok = -torch.min(ratio * a, clipped * a)
                    if ref_model is not None:
                        d = (
                            ref_logps[j] - logp
                        )  # k3 estimator of KL(policy || reference), always >= 0
                        kl = torch.exp(d) - d - 1
                        per_tok = per_tok + args.kl_beta * kl
                        if it == 0:
                            stats[5] += (kl.detach() * m).sum()
                    loss = (per_tok * m).sum() / n_tok
                    loss.backward()  # FSDP: re-gather each block, then reduce-scatter its gradients
                    if it > 0:
                        stats[6] += (
                            (((ratio - clipped).abs() > 0).float() * m).sum().detach()
                        )
                        stats[7] += m.sum()

                gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                if hasattr(
                    gnorm, "full_tensor"
                ):  # FSDP returns a DTensor; call on every rank
                    gnorm = gnorm.full_tensor()
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            sync()
            t_train += time.perf_counter() - t1
            step += 1

            stats[0] += rewards.sum()
            stats[1] += rewards.numel()
            stats[2] += no_signal
            stats[3] += rewards.numel() // G
            stats[4] += n_tok

            if step % args.log_every == 0 or step == total_steps:
                dist.all_reduce(stats)
                if rank == 0:
                    n = (
                        args.log_every
                        if step % args.log_every == 0
                        else step % args.log_every
                    )
                    kl = (
                        f"{stats[5].item() / stats[4].item():6.4f}"
                        if ref_model is not None
                        else "     -"
                    )
                    clip = (
                        f"{stats[6].item() / stats[7].item():5.1%}"
                        if stats[7] > 0
                        else "    -"
                    )
                    print(
                        f"{step:4d}/{total_steps:<4d}  {stats[0].item() / stats[1].item():6.3f}  "
                        f"{stats[2].item() / stats[3].item():9.0%}  {stats[4].item() / stats[1].item():4.1f}  "
                        f"{kl}  {clip}  {gnorm.item():6.2f}  {sched.get_last_lr()[0]:8.2e}  "
                        f"{t_roll / n:.1f}/{t_train / n:.1f}",
                        flush=True,
                    )
                    # One group of answers, to see what the model is actually being rewarded for.
                    group = [
                        f"{p.strip()!r}{'+' if r else '-'}"
                        for p, r in zip(preds[:G], rewards[:G].tolist())
                    ]
                    print(
                        f"           Q: {questions[0][:70]!r}  target {targets[0]!r}: {' '.join(group)}",
                        flush=True,
                    )
                stats.zero_()
                t_roll = t_train = 0.0
            if step >= total_steps:
                done = True
                break
        if done:
            break
    sync()
    train_s = time.perf_counter() - t_start

    peak = torch.tensor(
        [torch.cuda.max_memory_allocated(device) / 2**30 if use_cuda else 0.0],
        device=device,
    )
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    tok = torch.tensor([float(answer_tokens)], device=device)
    dist.all_reduce(tok)

    # 8. Accuracy after training.
    t0 = time.perf_counter()
    acc_after = evaluate(
        model,
        gen_model,
        eval_loader,
        processor,
        device,
        synced,
        args.max_new_tokens,
        rank,
    )
    if rank == 0:
        print(
            f"Accuracy after training: {acc_after:.1%} ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )

    row = {
        "mode": args.mode,
        "model": os.path.basename(os.path.normpath(args.model)),
        "gpus": world,
        "steps": total_steps,
        "questions_per_step": args.prompts_per_gpu * world,
        "G": G,
        "ppo_epochs": args.ppo_epochs,
        "kl_beta": args.kl_beta,
        "std_norm": not args.no_std_norm,
        "temperature": args.temperature,
        "lr": args.lr,
        "act_ckpt": args.ac,
        "train_vision": args.train_vision,
        "peak_gpu_mem_GB": round(peak.item(), 2),
        "answer_tokens": int(tok.item()),
        "train_min": round(train_s / 60, 1),
        "acc_before": "" if acc_before is None else round(acc_before, 4),
        "acc_after": round(acc_after, 4),
    }
    if rank == 0:
        print(row, flush=True)
        new_file = not os.path.exists(f"{args.exp_name}_results_grpo.csv")
        with open("results_grpo.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # 9. Save in Hugging Face format: gather the full weights onto rank 0's CPU, then save_pretrained.
    if args.save_dir:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        state = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        if rank == 0:
            gen_model.save_pretrained(args.save_dir, state_dict=state)
            processor.save_pretrained(args.save_dir)
            print(f"Saved to {args.save_dir}", flush=True)
        barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
