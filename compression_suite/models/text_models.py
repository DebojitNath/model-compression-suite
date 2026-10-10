"""
Transformer and text classification model architectures for NLP distillation and compression.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from compression_suite.models.base import BaseCompressibleModel
from compression_suite.models.registry import register_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]


class TransformerTextClassifier(BaseCompressibleModel):
    """
    Standard Transformer encoder text classifier with embedding,
    multi-head self-attention stack, and pooled classification head.
    """
    def __init__(
        self,
        vocab_size: int = 30522,
        num_classes: int = 2,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        
        mask = None
        if attention_mask is not None:
            # PyTorch transformer takes src_key_padding_mask where True indicates masked out elements
            mask = (attention_mask == 0)
            
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        # Pool first token ([CLS] token representation)
        cls_rep = encoded[:, 0, :]
        logits = self.classifier(cls_rep)
        return logits


@register_model("bert_teacher")
def build_bert_teacher(num_classes: int = 2, **kwargs) -> TransformerTextClassifier:
    return TransformerTextClassifier(
        vocab_size=kwargs.get("vocab_size", 30522),
        num_classes=num_classes,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_seq_len=kwargs.get("max_seq_len", 128),
    )


@register_model("minibert_student")
def build_minibert_student(num_classes: int = 2, **kwargs) -> TransformerTextClassifier:
    return TransformerTextClassifier(
        vocab_size=kwargs.get("vocab_size", 30522),
        num_classes=num_classes,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        max_seq_len=kwargs.get("max_seq_len", 128),
    )


@register_model("fast_transformer_teacher")
def build_fast_transformer_teacher(num_classes: int = 2, **kwargs) -> TransformerTextClassifier:
    return TransformerTextClassifier(
        vocab_size=1000,
        num_classes=num_classes,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        max_seq_len=32,
    )


@register_model("fast_transformer_student")
def build_fast_transformer_student(num_classes: int = 2, **kwargs) -> TransformerTextClassifier:
    return TransformerTextClassifier(
        vocab_size=1000,
        num_classes=num_classes,
        d_model=64,
        nhead=2,
        num_layers=1,
        dim_feedforward=128,
        max_seq_len=32,
    )
