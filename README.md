# DTAI Parallel CPU Streaming

This package provides CPU-master streaming for PyTorch models launched with
`torchrun`. The public API is an in-place transformation: choose an ordered
submodule such as `layers` or `decoder.layers`, replace it with streaming stage
wrappers, and train through the engine returned by `apply_cpu_streaming_`.

The transformed model keeps its normal Python structure. Surrounding modules such
as embeddings, norms, heads, and resident decoder blocks remain ordinary PyTorch
modules. The engine wraps the model in `DistributedDataParallel`, so resident
parameters follow PyTorch's standard distributed path.

Offloaded stages keep their true parameters, buffers, gradients, and
optimizer-state tensor leaves as file-backed CPU masters under `/dev/shm`. During
forward and backward, each stage streams temporary tensor copies to the
process-local device. Backward replays the stage with gradient-tracking parameter
copies, the engine averages offloaded gradients across ranks, and the owner rank
for each stage runs the PyTorch optimizer update before the next step.

## Training Loop

Training code launches under `torchrun`, then uses the engine as the optimizer
and distributed coordination object. The forward and backward pass still run
through a normal PyTorch module interface exposed as `engine.model`.

```python
import torch
from torch import nn

from dtai_parallel import apply_cpu_streaming_


class DecoderBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, hidden, *, attention_mask=None):
        return hidden + self.net(hidden)


class Model(nn.Module):
    def __init__(self, block_factory, vocab_size, width, depth):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.layers = nn.ModuleList([block_factory(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)

    def forward(self, tokens, *, attention_mask=None):
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)
        return self.head(self.norm(hidden))


model = Model(block_factory=DecoderBlock, vocab_size=32000, width=4096, depth=32)
engine = apply_cpu_streaming_(
    model,
    "layers",
    offload_policy=True,
    resident_suffix_count=2,
    optimizer_cls=torch.optim.AdamW,
    optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.01, "foreach": False},
    max_grad_norm=1.0,
)

criterion = nn.CrossEntropyLoss()

for batch in loader:
    tokens = batch["tokens"].to(engine.local_device)
    labels = batch["labels"].to(engine.local_device)
    attention_mask = batch["attention_mask"].to(engine.local_device)

    engine.zero_grad(set_to_none=True)
    logits = engine.model(tokens, attention_mask=attention_mask)
    loss = criterion(logits.flatten(0, 1), labels.flatten())
    loss.backward()
    engine.step()

final_model = engine.close()
```

The returned `final_model` is an ordinary PyTorch model on the close rank. Its
transformed submodule has been materialized back into a normal `ModuleList` or
`Sequential`.

## Offload Policy

`offload_policy=True` streams every item in the selected ordered submodule.
`offload_policy=False` keeps every item resident. A boolean sequence controls
individual stages, and a callable can make the decision from the stage index,
name, and module.

```python
engine = apply_cpu_streaming_(
    model,
    "decoder.layers",
    offload_policy=lambda index, name, module: index % 2 == 0,
    optimizer_cls=torch.optim.AdamW,
    optimizer_kwargs={"lr": 3e-4, "foreach": False},
)
```

`resident_suffix_count` keeps the final stages resident after the offload policy
has been evaluated. This is useful for decoder stacks that benefit from keeping
the last few blocks on device.

## Tests

Run the full test suite with:

```bash
uv run pytest -q
```

The tests cover result equivalence, torchrun launch behavior, shared CPU storage,
mixed resident and offloaded stages, nested module paths, arbitrary tensor
arguments, nested tensor outputs, transfer timing, and memory benchmark
invariants.

## Benchmarks

Run all benchmark entrypoints with:

```bash
uv run python -m benchmarks.streaming_memory run --mode one-gpu --mode two-gpu && uv run python -m benchmarks.qwen_memory run --mode one-gpu --mode two-gpu
```

The benchmark CLIs launch their measured cases through `torchrun`, print a
`tqdm` progress bar over the case list, write per-case JSON payloads, and write a
`summary.json` file under `benchmark-results/`.
