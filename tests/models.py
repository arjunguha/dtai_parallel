from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten

DIM = 8
ZOO = ["mlp", "container", "dropout", "buffer", "frozen", "transformer", "kwargs"]


class MLP(nn.Module):
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
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, state: dict) -> dict:
        x = torch.relu(self.lin(state["x"]) + state["bias"])
        return {"x": x, "bias": state["bias"], "depth": state["depth"] + 1}


class ContainerModel(nn.Module):
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
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(torch.relu(self.lin(x)))


class DropoutModel(nn.Module):
    def __init__(self, dim: int = DIM, depth: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList(DropoutBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class BufferBlock(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.register_buffer("scale", torch.rand(dim) + 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * self.scale


class BufferModel(nn.Module):
    def __init__(self, dim: int = DIM, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(BufferBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class FrozenBlock(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.frozen = nn.Parameter(torch.randn(dim, dim) / dim)
        self.frozen.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x) + x @ self.frozen)


class FrozenModel(nn.Module):
    def __init__(self, dim: int = DIM, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(FrozenBlock(dim) for _ in range(depth))
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class DecoderLayer(nn.Module):
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


def make_model(name: str) -> nn.Module:
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
    if name in {"transformer", "kwargs"}:
        return "decoder.layers"
    return "blocks"


def make_input(name: str, batch: int, device: torch.device, *, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if name in {"transformer", "kwargs"}:
        idx = torch.randint(0, 16, (batch, 4), generator=generator).to(device)
        if name == "kwargs":
            mask = torch.randn(batch, 4, generator=generator, device="cpu").to(device)
            return (idx, mask)
        return idx
    return torch.randn(batch, DIM, generator=generator).to(device)


def call_model(model: nn.Module, name: str, batch):
    if name == "kwargs":
        idx, mask = batch
        return model(idx, attention_mask=mask, scale=0.75)
    return model(batch)


def loss_of(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return (out ** 2).sum()
    leaves, _ = tree_flatten(out)
    tensor_terms = [(leaf ** 2).sum() for leaf in leaves if isinstance(leaf, torch.Tensor) and leaf.requires_grad]
    if not tensor_terms:
        tensor_terms = [(leaf ** 2).sum() for leaf in leaves if isinstance(leaf, torch.Tensor)]
    return sum(tensor_terms)
