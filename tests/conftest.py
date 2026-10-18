"""Shared fixtures and test mocks for code-embed test suite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig


class DummyBackboneConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self, hidden_size: int = 64, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class DummyTransformerBackbone(nn.Module):
    """Small transformer backbone mock for fast, isolated testing without downloading weights."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.config = DummyBackboneConfig(hidden_size=hidden_size)
        self.embedding = nn.Embedding(1000, hidden_size)
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        emb = self.embedding(input_ids)
        out = self.layer(emb)

        class Output:
            def __init__(self, last_hidden_state: torch.Tensor) -> None:
                self.last_hidden_state = last_hidden_state

        return Output(out)

    def save_pretrained(self, save_directory: str) -> None:
        pass


class DummyTokenizer:
    """Lightweight dummy tokenizer for testing collators and models."""

    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.eos_token_id = 1
        self.unk_token = "<unk>"
        self.unk_token_id = 2
        self.vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "def": 3, "return": 4}
        self.model_max_length = 512

    def __len__(self) -> int:
        return len(self.vocab)

    def convert_ids_to_tokens(self, ids: Any, skip_special_tokens: bool = False) -> Any:
        if isinstance(ids, int):
            return "<pad>"
        return ["<pad>"] * len(ids)

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(text, str):
            text = [text]

        batch_size = len(text)
        seq_len = 16

        input_ids = torch.randint(1, 50, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        if return_tensors == "np":
            return {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
            }

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self.vocab.get(w, 2) for w in text.split()]

    def save_pretrained(self, save_directory: str) -> tuple[str, ...]:
        return ()


@pytest.fixture
def dummy_backbone() -> nn.Module:
    return DummyTransformerBackbone(hidden_size=64)


@pytest.fixture
def dummy_tokenizer() -> Any:
    return DummyTokenizer()


@pytest.fixture
def sample_code_snippets() -> list[str]:
    return [
        "def add(a: int, b: int) -> int:\n    return a + b",
        'def subtract(x: float, y: float) -> float:\n    """Return difference."""\n    return x - y',
        "class Calculator:\n    def multiply(self, a, b):\n        return a * b",
    ]


@pytest.fixture
def sample_queries() -> list[str]:
    return [
        "add two numbers",
        "calculate subtraction",
        "multiply two numbers",
    ]


@pytest.fixture
def random_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    vecs = rng.normal(size=(20, 32)).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms
