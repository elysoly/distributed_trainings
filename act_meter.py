import collections
import re

import torch
import torch.distributed as dist


class SavedActivationMeter:
    """Bytes that autograd saves for backward during one forward pass = activation memory.

    Use as a context manager around a real training forward (train mode, grad enabled).
    Each storage is counted once; parameters are skipped (weights are not activations).
    Bytes are attributed to the deepest module (up to `max_dots` levels) running at that moment.
    """

    def __init__(self, model, max_dots=3):
        self.model = getattr(model, "module", model)  # unwrap DDP
        self.max_dots = max_dots
        self.stack, self.seen, self.handles = [], set(), []
        self.by_module = collections.Counter()

    def _pack(self, t):
        if isinstance(t, torch.nn.Parameter):
            return t
        storage = t.untyped_storage()
        if storage.data_ptr() not in self.seen:  # many ops save the same tensor
            self.seen.add(storage.data_ptr())
            self.by_module[self.stack[-1] if self.stack else ""] += storage.nbytes()
        return t

    def __enter__(self):
        def push(name):
            def hook(module, args):
                self.stack.append(
                    name
                )  # returns None: a pre-hook's return value would replace the inputs

            return hook

        def pop(module, args, output):
            self.stack.pop()  # don't return it: a forward hook's return value replaces the output

        for name, m in self.model.named_modules():
            if name.count(".") > self.max_dots:
                continue
            self.handles.append(m.register_forward_pre_hook(push(name)))
            self.handles.append(m.register_forward_hook(pop))
        self._ctx = torch.autograd.graph.saved_tensors_hooks(self._pack, lambda t: t)
        self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        for h in self.handles:
            h.remove()

    def report(self, device, rank, fwd_delta_bytes=None, top=8):
        """Collective: call on every rank. Prints rank 0's breakdown and the max over ranks."""
        mine = sum(self.by_module.values())
        vals = torch.tensor(
            [mine, fwd_delta_bytes or 0], dtype=torch.float64, device=device
        )
        if dist.is_initialized():
            dist.all_reduce(vals, op=dist.ReduceOp.MAX)
        if rank == 0:
            g = 2**30
            print(
                f"Saved activations: rank0 {mine / g:.2f} GiB, max over ranks {vals[0].item() / g:.2f} GiB"
            )
            if fwd_delta_bytes is not None:
                print(
                    f"memory_allocated growth during forward: max over ranks {vals[1].item() / g:.2f} GiB"
                )
            groups = collections.Counter()
            for name, b in self.by_module.items():
                groups[re.sub(r"\.\d+", ".*", name) or "<top level: loss>"] += b
            for name, b in groups.most_common(top):
                print(f"  {b / g:7.3f} GiB  {b / mine:5.1%}  {name}")
