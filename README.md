# CPU-master streaming for a transformed ModuleList

This is a compact PyTorch reference implementation of a GH200-oriented training abstraction: keep offloaded decoder-block parameters, gradients, and optimizer state as CPU masters, and stream one layer at a time to the process-local device.  The API is an in-place transformation on an existing model, not a replacement base class.

```python
engine = apply_cpu_streaming_(model, "decoder.layers", ...)
```

The model remains the model the user wrote.  The transformation replaces only the selected `nn.ModuleList` or `nn.Sequential`; surrounding modules such as embeddings, norms, and unembeddings remain ordinary resident modules.

## Intended torchrun shape

The implementation assumes the common `torchrun` layout: one process owns one GPU.  Each process forwards only its local batch.  The training loop is responsible for moving input tensors to the process-local device.

When a distributed process group is active and `wrap_ddp=True`, the engine wraps the transformed model in `torch.nn.parallel.DistributedDataParallel`.  This automatically handles resident parameters.  Offloaded parameters are hidden from the module tree, so DDP does not see them; the engine explicitly averages their gradients and dispatches their optimizer steps.

## What is streamed

For an offloaded stage, the CPU master module owns the true parameters and buffers.  Forward prefetches the stage's parameter and buffer state to the local device, calls the layer with the user's original `*args` and `**kwargs`, saves activations, and drops the temporary streamed state.  Backward replays the same layer under autograd with streamed parameter copies that require gradients, accumulates local parameter gradients into the CPU master, and then `engine.step()` averages them across ranks.

The optimizer path does not implement AdamW or SGD equations.  For each offloaded stage, the engine constructs the requested PyTorch optimizer class on temporary device parameters, restores that stage's opaque optimizer state, calls `optimizer.step()`, and writes the updated weights and optimizer state back to CPU.  Optimizer state for offloaded layers is sharded by stage owner rank.  Resident parameters use one ordinary PyTorch optimizer attached to DDP-visible parameters.

Asynchronous prefetching is implemented for CUDA with side streams and events.  During forward, a stage schedules the next offloaded stage.  During backward, a stage schedules the previous offloaded stage.  On CPU the same interface is used, but prefetching is necessarily synchronous.

## Minimal usage

```python
import torch
from torch import nn
from cpu_streaming_ddp import apply_cpu_streaming_

class Decoder(nn.Module):
    def __init__(self, block_factory, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([block_factory() for _ in range(n_layers)])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32000, 4096)
        self.decoder = Decoder(lambda: MyDecoderBlock(), n_layers=32)
        self.norm = nn.LayerNorm(4096)
        self.lm_head = nn.Linear(4096, 32000, bias=False)

    def forward(self, tokens, *, attention_mask=None, position_ids=None):
        x = self.embed(tokens)
        for layer in self.decoder.layers:
            x = layer(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        return self.lm_head(self.norm(x))

model = Model()
engine = apply_cpu_streaming_(
    model,
    "decoder.layers",
    offload_policy=True,
    optimizer_cls=torch.optim.AdamW,
    optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.01, "foreach": False},
    max_grad_norm=1.0,
)

criterion = nn.CrossEntropyLoss()
for tokens, labels, attention_mask in loader:
    tokens = tokens.to(engine.local_device)
    labels = labels.to(engine.local_device)
    attention_mask = attention_mask.to(engine.local_device)

    engine.zero_grad(set_to_none=True)
    logits = engine.model(tokens, attention_mask=attention_mask)
    loss = criterion(logits.flatten(0, 1), labels.flatten())
    loss.backward()
    total_norm = engine.step()

ordinary_model = engine.close()
```

`ordinary_model` is returned only on rank 0 by default.  Use `engine.close(return_on_all_ranks=True)` when every rank should receive a materialized copy.  The returned model contains a normal `nn.ModuleList` or `nn.Sequential`, not streaming wrappers.

## Mixed resident and offloaded layers

`offload_policy` can be a single boolean, a sequence of booleans, or a callable receiving `(index, name, module)`.  For example, this offloads every other decoder block while leaving the rest resident and DDP-managed:

```python
engine = apply_cpu_streaming_(
    model,
    "decoder.layers",
    offload_policy=lambda i, name, module: i % 2 == 0,
    optimizer_cls=torch.optim.AdamW,
    optimizer_kwargs={"lr": 3e-4, "foreach": False},
)
```

## Tests

Run the suite with:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

In this CPU-only environment the CUDA cases skip automatically.  The suite includes single-process equivalence tests, mixed resident/offloaded tests, nested `decoder.layers` transformation tests, arbitrary `*args`/`**kwargs` and nested-output tests, prefetch-scheduling tests, and a real two-process DDP equivalence check using Gloo on CPU or NCCL on CUDA.
