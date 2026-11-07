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
from act_meter import SavedActivationMeter
from datasets import load_dataset
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import fully_shard
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
    p.add_argument("--mode", choices=["ddp", "fsdp"], required=True)
    p.add_argument(
        "--model", default="Qwen/Qwen3-VL-2B-Instruct", help="Hub name or local folder"
    )
    p.add_argument(
        "--data", default="HuggingFaceM4/ChartQA", help="Hub name or local folder"
    )
    p.add_argument("--train-samples", type=int, default=2000)
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="longest image side in pixels; ~300 image tokens at 640 for Qwen3-VL",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--batch-size", type=int, default=2, help="examples per GPU per micro-step"
    )
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--max-steps", type=int, default=0, help="stop early (0 = run all epochs)"
    )
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--ac", action="store_true", help="activation checkpointing on every block"
    )
    p.add_argument(
        "--train-vision", action="store_true", help="also train the vision encoder"
    )
    p.add_argument(
        "--freeze-embeddings", action="store_true", help="freeze embedding layers"
    )

    p.add_argument(
        "--skip-baseline", action="store_true", help="skip the eval before training"
    )
    p.add_argument(
        "--save-dir",
        default=None,
        help="save the fine-tuned model in Hugging Face format",
    )
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exp-name", type=str, default="exp_vl_")

    p.add_argument(
        "--mem-snapshot",
        action="store_true",
        help="dump a memory snapshot after 3 steps",
    )

    p.add_argument(
        "--activation-measure",
        action="store_true",
        help="compute activation memory usage",
    )
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


def answer_start(ids, header):
    """Index of the first answer token: right after the last '<|im_start|>assistant\\n'."""
    h = len(header)
    for i in range(len(ids) - h, -1, -1):
        if ids[i : i + h] == header:
            return i + h
    raise ValueError(
        "Assistant header not found; this script expects a Qwen (ChatML) chat template."
    )


def collate_train(examples, processor, max_side, header):
    processor.tokenizer.padding_side = "right"
    texts, images = [], []
    for ex in examples:
        messages = [
            user_turn(ex["query"]),
            {"role": "assistant", "content": answer_of(ex)},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False))
        images.append(prep_image(ex["image"], max_side))
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")

    # Train only on the answer: mask the prompt (incl. image tokens) and the padding.
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    for row in range(labels.shape[0]):
        labels[row, : answer_start(batch["input_ids"][row].tolist(), header)] = -100
    batch["labels"] = labels
    return batch


def collate_eval(examples, processor, max_side):
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


@torch.no_grad()
def evaluate(model, gen_model, loader, processor, device, synced, max_new_tokens, rank):
    model.eval()
    correct = total = 0
    shown = 0
    for batch, targets, questions in loader:
        batch = batch.to(device)
        # synced_gpus: under FSDP every forward is a collective, so all ranks must keep
        # stepping until the slowest rank has finished, or the all-gathers deadlock.
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


# ----------------------------------------------------------------------------- main


def main():
    args = parse_args()

    use_cuda = torch.cuda.is_available()  # CPU + gloo works too, for debugging only
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        if args.mem_snapshot and rank == 0:
            torch.cuda.memory._record_memory_history(max_entries=200000, stacks="all")

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
    header = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )

    # Sanity check: which tokens of one example are trained on?
    if rank == 0:
        b = collate_train([train_ds[0]], processor, args.image_size, header)
        trained = b["labels"][0][b["labels"][0] != -100]
        print(
            f"Example 0: {b['input_ids'].shape[1]} tokens; trained on "
            f"{processor.tokenizer.decode(trained)!r}",
            flush=True,
        )

    # 2. Model, full weights in fp32 on CPU.
    #    fp32 on purpose: TITAN RTX (Turing) has no bf16, and fp16 needs loss scaling.
    model = AutoVLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa"
    )

    if not args.train_vision:
        for name, param in model.named_parameters():
            if "visual" in name and "merger" not in name:
                param.requires_grad_(False)
    if args.freeze_embeddings:
        model.get_input_embeddings().weight.requires_grad_(False)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Transformer blocks = the classes HF says must never be split across devices.
    block_names = set(getattr(model, "_no_split_modules", None) or [])
    if args.ac:
        # Wrap each block in a checkpoint wrapper BEFORE sharding, so FSDP's hooks sit
        # outside the checkpoint. (HF's gradient_checkpointing_enable() puts the
        # checkpoint outside FSDP's hooks and causes an extra all-gather in backward.)
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

    # 3. Distribute the model.
    if args.mode == "fsdp":
        # One FSDP unit per block (vision blocks and decoder layers), then the root.
        for block in blocks:
            fully_shard(block)
        fully_shard(model)
        gen_model = model
    else:
        model.to(device)
        model = DDP(model, device_ids=[local_rank] if use_cuda else None)
        gen_model = (
            model.module
        )  # generate() on the plain module; DDP only wraps forward

    if rank == 0:
        kinds = sorted(
            {
                type(
                    b._checkpoint_wrapped_module
                    if isinstance(b, CheckpointWrapper)
                    else b
                ).__name__
                for b in blocks
            }
        )
        print(
            f"[{args.mode}] {n_total / 1e9:.2f}B params, {n_train / 1e9:.2f}B trainable; "
            f"{len(blocks)} blocks ({', '.join(kinds)})",
            flush=True,
        )

    # 4. Data loaders. Every rank must run the same number of steps: under FSDP each
    #    forward is a collective, so a rank with one extra batch would hang the others.
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
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=2,
        collate_fn=functools.partial(
            collate_train, processor=processor, max_side=args.image_size, header=header
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
            collate_eval, processor=processor, max_side=args.image_size
        ),
    )
    synced = args.mode == "fsdp" and world > 1

    # 5. Optimizer on trainable params, created AFTER sharding (so it holds the shards).
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    if total_steps == 0:
        raise ValueError(
            "Not enough training samples for GPUs x batch size x grad accum"
        )
    warmup = max(1, int(args.warmup_ratio * total_steps))

    def lr_lambda(step):  # linear warmup, then cosine decay to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))
        )

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # 6. Baseline accuracy.
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

    # 7. Train.
    if rank == 0:
        print(
            f"Training {total_steps} steps; global batch = {args.batch_size} x {world} GPUs "
            f"x {args.grad_accum} accum = {args.batch_size * world * args.grad_accum}",
            flush=True,
        )
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step, tokens = 0, 0
    step_loss = torch.zeros((), device=device)
    sync()
    t_start = t_log = time.perf_counter()
    done = False
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch * args.grad_accum:
                break
            tokens += int(
                batch["attention_mask"].sum()
            )  # text + image tokens, no padding
            batch = batch.to(device)

            if args.activation_measure:
                measure = (
                    epoch == 0 and i == 0
                )  # first micro-step only: same condition on every rank
                mem0 = torch.cuda.memory_allocated(device)
                with (
                    SavedActivationMeter(model)
                    if measure
                    else contextlib.nullcontext() as meter
                ):
                    loss = model(**batch, use_cache=False).loss / args.grad_accum
                if measure:
                    meter.report(
                        device,
                        rank,
                        fwd_delta_bytes=torch.cuda.memory_allocated(device) - mem0,
                    )
            else:
                loss = model(**batch, use_cache=False).loss / args.grad_accum

            loss.backward()  # FSDP: re-gather each block, then reduce-scatter its gradients
            step_loss += loss.detach()
            if (i + 1) % args.grad_accum:
                continue  # keep accumulating (gradients are still reduced every micro-step)

            gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if hasattr(
                gnorm, "full_tensor"
            ):  # FSDP returns a DTensor; call on every rank
                gnorm = gnorm.full_tensor()
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                dist.all_reduce(step_loss)
                sync()
                now = time.perf_counter()
                if rank == 0:
                    n = (
                        args.log_every
                        if step % args.log_every == 0
                        else step % args.log_every
                    )
                    print(
                        f"step {step:4d}/{total_steps}  loss {step_loss.item() / world:.4f}  "
                        f"grad_norm {gnorm.item():.2f}  lr {sched.get_last_lr()[0]:.2e}  "
                        f"{(now - t_log) / n:.2f} s/step",
                        flush=True,
                    )
                t_log = now
            if args.mem_snapshot and use_cuda and rank == 0 and step == 3:
                print("Dumping memory snapshot to mem_train_rank0.pickle", flush=True)
                torch.cuda.memory._dump_snapshot("mem_train_rank0.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)  # stop recording

            step_loss.zero_()
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
    tok = torch.tensor([float(tokens)], device=device)
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
            f"Accuracy after training: {acc_after:.1%} "
            f"({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )

    row = {
        "mode": args.mode,
        "model": os.path.basename(os.path.normpath(args.model)),
        "gpus": world,
        "train_samples": len(train_ds),
        "steps": total_steps,
        "global_batch": args.batch_size * world * args.grad_accum,
        "batch_per_gpu": args.batch_size,
        "grad_accum": args.grad_accum,
        "image_size": args.image_size,
        "lr": args.lr,
        "act_ckpt": args.ac,
        "train_vision": args.train_vision,
        "trainable_B": round(n_train / 1e9, 2),
        "peak_gpu_mem_GB": round(peak.item(), 2),
        "train_tokens_per_s": round(tok.item() / train_s),
        "train_min": round(train_s / 60, 1),
        "acc_before": "" if acc_before is None else round(acc_before, 4),
        "acc_after": round(acc_after, 4),
    }
    if rank == 0:
        print(row, flush=True)
        new_file = not os.path.exists("results_vl.csv")
        with open(f"{args.exp_name}_results_vl.csv", "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # 9. Save in Hugging Face format: gather the full weights onto rank 0's CPU, then
    #    save_pretrained. The result loads like any HF model (vLLM, Unsloth, your GRPO run).
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
