from __future__ import annotations

import torch
import torch.nn as nn
from typing import Mapping, Optional

from torch import Tensor
from torch.utils._pytree import tree_flatten

DIM = 8
STREAM_INPUT_DIM = 128
STREAM_OUTPUT_DIM = 128
ZOO = ["mlp", "container", "dropout", "buffer", "frozen", "transformer", "kwargs"]


class MLP(nn.Module):
    """Simple ModuleList MLP used by generic model-zoo checks."""

    def __init__(self, dim: int = DIM, depth: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, dim), nn.GELU()) for _ in range(depth)
        )
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class ContainerBlock(nn.Module):
    """Block that passes structured Python containers through streaming boundaries."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, state: dict) -> dict:
        x = torch.relu(self.lin(state["x"]) + state["bias"])
        return {"x": x, "bias": state["bias"], "depth": state["depth"] + 1}


class ContainerModel(nn.Module):
    """Model-zoo case that keeps tensors inside dictionaries between blocks."""

    def __init__(self, dim: int = DIM, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(ContainerBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = {"x": x, "bias": torch.ones_like(x), "depth": 0}
        for block in self.blocks:
            state = block(state)
        assert state["depth"] == len(self.blocks)
        return self.head(state["x"])


class DropoutBlock(nn.Module):
    """Block with dropout for RNG preservation coverage."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.relu(self.lin(x)))


class DropoutModel(nn.Module):
    """Model-zoo case that exercises dropout under streamed replay."""

    def __init__(self, dim: int = DIM, depth: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(DropoutBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class BufferBlock(nn.Module):
    """Block with a registered buffer that must stream with parameters."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.register_buffer("scale", torch.rand(dim) + 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * self.scale


class BufferModel(nn.Module):
    """Model-zoo case that verifies buffers participate in streamed state."""

    def __init__(self, dim: int = DIM, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(BufferBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class FrozenBlock(nn.Module):
    """Block containing a frozen parameter alongside trainable parameters."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.frozen = nn.Parameter(torch.randn(dim, dim) / dim)
        self.frozen.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x) + x @ self.frozen)


class FrozenModel(nn.Module):
    """Model-zoo case for preserving frozen parameters through transformation."""

    def __init__(self, dim: int = DIM, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(FrozenBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class DecoderLayer(nn.Module):
    """Tiny transformer decoder layer for nested decoder path coverage."""

    def __init__(self, dim: int = DIM, heads: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class MiniTransformer(nn.Module):
    """Small transformer-shaped model with decoder.layers as the target path."""

    def __init__(self, vocab: int = 16, dim: int = DIM, heads: int = 2, depth: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(DecoderLayer(dim, heads) for _ in range(depth))
        self.lnf = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for layer in self.decoder.layers:
            x = layer(x)
        return self.head(self.lnf(x))


class KwargDecoderLayer(nn.Module):
    """Decoder layer whose forward uses kwargs and structured outputs."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        residual: torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> dict:
        del metadata
        y = self.proj(self.norm(hidden))
        if attention_mask is not None:
            y = y + attention_mask.unsqueeze(-1).to(dtype=y.dtype)
        if residual is not None:
            y = y + 0.1 * residual
        hidden = hidden + scale * torch.tanh(y)
        return {
            "hidden": hidden,
            "aux": (hidden.detach(), {"depth_tag": "kept-as-python"}),
        }


class KwargSandwich(nn.Module):
    """Model-zoo case that returns nested structures from a kwarg-heavy decoder."""

    def __init__(self, vocab: int = 16, dim: int = DIM, depth: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList(KwargDecoderLayer(dim) for _ in range(depth))
        self.unembed = nn.Linear(dim, vocab)

    def forward(self, idx: torch.Tensor, attention_mask: torch.Tensor | None = None, *, scale: float = 1.0) -> dict:
        hidden = self.embed(idx)
        for layer in self.decoder.layers:
            state = layer(
                hidden,
                attention_mask,
                scale=scale,
                residual=hidden,
                metadata={"layer_kind": "decoder"},
            )
            hidden = state["hidden"]
        return {"logits": self.unembed(hidden), "final_hidden": hidden.detach()}


class ResidualBlock(nn.Module):
    """Small residual MLP block used by equivalence tests."""

    def __init__(self, width: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.lin1 = nn.Linear(width, width, dtype=dtype)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(width, width, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.25 * self.lin2(self.act(self.lin1(x)))


class SandwichModel(nn.Module):
    """Embedding and unembedding around a ModuleList of decoder-ish blocks."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.embed = nn.Linear(5, 8, dtype=dtype)
        self.layers = nn.ModuleList([ResidualBlock(8, dtype=dtype) for _ in range(3)])
        self.norm = nn.LayerNorm(8, dtype=dtype)
        self.unembed = nn.Linear(8, 3, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.unembed(x)


class SequentialSandwich(nn.Module):
    """Sequential-container variant of the small sandwich model."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.in_proj = nn.Linear(5, 8, dtype=dtype)
        self.blocks = nn.Sequential(
            ResidualBlock(8, dtype=dtype),
            ResidualBlock(8, dtype=dtype),
            ResidualBlock(8, dtype=dtype),
        )
        self.out_proj = nn.Linear(8, 3, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(self.blocks(self.in_proj(x)))


class KwargDecoderBlock(nn.Module):
    """A block whose forward has multiple args, kwargs, and nested outputs."""

    def __init__(self, width: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, width, dtype=dtype)
        self.mix = nn.Linear(width, width, dtype=dtype)
        self.norm = nn.LayerNorm(width, dtype=dtype)

    def forward(
        self,
        hidden: Tensor,
        mask: Tensor,
        additive: Optional[Tensor] = None,
        *,
        scale: float = 1.0,
        metadata: Optional[Mapping[str, object]] = None,
        return_aux: bool = True,
    ):
        del metadata
        update = self.mix(torch.tanh(self.in_proj(hidden))) * mask
        if additive is not None:
            update = update + additive
        out = self.norm(hidden + float(scale) * update)
        aux = {
            "mean": out.mean(dim=-1),
            "positive_mask": mask > 0.0,
        }
        return (out, aux) if return_aux else out


class NestedDecoder(nn.Module):
    """Container for nested decoder.layers transformation tests."""

    def __init__(self, width: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.layers = nn.ModuleList([KwargDecoderBlock(width, dtype=dtype) for _ in range(2)])


class KwargSandwichModel(nn.Module):
    """Persistent resident modules around a nested decoder ModuleList."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.embed = nn.Linear(5, 8, dtype=dtype)
        self.context = nn.Linear(4, 8, dtype=dtype)
        self.decoder = NestedDecoder(8, dtype=dtype)
        self.unembed = nn.Linear(8, 3, dtype=dtype)

    def forward(self, x: Tensor, mask: Tensor, context: Tensor, *, scale: float = 0.5) -> Tensor:
        hidden = self.embed(x)
        additive = 0.05 * self.context(context)
        aux_penalty = hidden.new_zeros(())
        for index, layer in enumerate(self.decoder.layers):
            hidden, aux = layer(
                hidden,
                mask,
                additive,
                scale=scale,
                metadata={"layer_index": index, "note": "non-tensor kwargs are replay constants"},
                return_aux=True,
            )
            aux_penalty = aux_penalty + aux["mean"].mean()
        return self.unembed(hidden) + 1e-3 * aux_penalty


class LargeResidualBlock(nn.Module):
    """Large residual MLP block used by memory benchmark models."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        return x + 0.25 * self.lin2(self.act(self.lin1(x)))


class LargeStreamingSandwich(nn.Module):
    """Embedding and head around a large streamed ModuleList."""

    def __init__(self, num_layers: int, layer_width: int) -> None:
        super().__init__()
        self.embed = nn.Linear(STREAM_INPUT_DIM, layer_width)
        self.layers = nn.ModuleList(LargeResidualBlock(layer_width) for _ in range(num_layers))
        self.norm = nn.LayerNorm(layer_width)
        self.unembed = nn.Linear(layer_width, STREAM_OUTPUT_DIM)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.unembed(self.norm(x))


def make_model(name: str) -> nn.Module:
    """Construct one named model-zoo case for generic streaming tests."""
    if name == "mlp":
        return MLP()
    if name == "container":
        return ContainerModel()
    if name == "dropout":
        return DropoutModel()
    if name == "buffer":
        return BufferModel()
    if name == "frozen":
        return FrozenModel()
    if name == "transformer":
        return MiniTransformer()
    if name == "kwargs":
        return KwargSandwich()
    raise KeyError(name)


def target_path(name: str) -> str:
    """Return the module path transformed for a named model-zoo case."""
    if name in {"transformer", "kwargs"}:
        return "decoder.layers"
    return "blocks"


def make_input(name: str, batch: int, device: torch.device, *, seed: int):
    """Create deterministic input matching a named model-zoo case."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if name in {"transformer", "kwargs"}:
        idx = torch.randint(0, 16, (batch, 4), generator=generator).to(device)
        if name == "kwargs":
            mask = torch.randn(batch, 4, generator=generator, device="cpu").to(device)
            return (idx, mask)
        return idx
    return torch.randn(batch, DIM, generator=generator).to(device)


def call_model(model: nn.Module, name: str, batch):
    """Call a model-zoo case with the right positional and keyword shape."""
    if name == "kwargs":
        idx, mask = batch
        return model(idx, attention_mask=mask, scale=0.75)
    return model(batch)


def loss_of(out) -> torch.Tensor:
    """Reduce tensor or structured model outputs to a scalar training loss."""
    if isinstance(out, torch.Tensor):
        return (out ** 2).sum()
    leaves, _ = tree_flatten(out)
    tensor_terms = [(leaf ** 2).sum() for leaf in leaves if isinstance(leaf, torch.Tensor) and leaf.requires_grad]
    if not tensor_terms:
        tensor_terms = [(leaf ** 2).sum() for leaf in leaves if isinstance(leaf, torch.Tensor)]
    return sum(tensor_terms)
