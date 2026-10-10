import pytest
import torch
from compression_suite.data.datasets import (
    get_dataloaders,
    tokenize_sentence,
    SST2Dataset,
    SyntheticVisionDataset,
    SyntheticTextDataset,
    infer_sample_input,
)
from compression_suite.models.registry import build_model


def test_synthetic_vision_and_text_datasets():
    vis = SyntheticVisionDataset(num_samples=50, in_channels=3, height=32, width=32, num_classes=10)
    assert len(vis) == 50
    img, target = vis[0]
    assert img.shape == (3, 32, 32)
    assert 0 <= target.item() < 10

    txt = SyntheticTextDataset(num_samples=50, seq_len=32, vocab_size=1000, num_classes=2)
    assert len(txt) == 50
    tokens, target = txt[0]
    assert tokens.shape == (32,)
    assert 0 <= target.item() < 2


def test_tokenizer_sentence_deterministic():
    sentence = "A brilliant and captivating film with great cinematography ."
    tokens = tokenize_sentence(sentence, max_seq_len=64, vocab_size=30522)
    assert tokens.shape == (64,)
    assert tokens[0].item() == 101  # [CLS]
    # Check that it ends with [SEP] before padding
    non_zero = tokens[tokens != 0]
    assert non_zero[-1].item() == 102  # [SEP]

    # Determinism across calls
    tokens2 = tokenize_sentence(sentence, max_seq_len=64, vocab_size=30522)
    assert torch.equal(tokens, tokens2)


def test_sst2_dataset_and_dataloaders():
    # Test SST-2 loader (uses cached or fallback data gracefully)
    ds = SST2Dataset(split="train", max_samples=40, download=False)
    assert len(ds) >= 20
    tokens, label = ds[0]
    assert tokens.shape == (128,)
    assert label.item() in (0, 1)

    loaders = get_dataloaders(dataset_name="sst2", batch_size=16)
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders
    assert "calib" in loaders

    sample = infer_sample_input(loaders)
    assert sample.shape[0] == 1
    assert sample.shape[1] == 128

    # Forward through minibert student
    student = build_model("minibert_student", num_classes=2)
    out = student(sample)
    assert out.shape == (1, 2)


def test_synthetic_dataloaders_factory():
    loaders_vis = get_dataloaders(dataset_name="synthetic", batch_size=32)
    sample_vis = infer_sample_input(loaders_vis)
    assert sample_vis.shape == (1, 3, 32, 32)

    loaders_txt = get_dataloaders(dataset_name="synthetic_text", batch_size=32, num_classes=2)
    sample_txt = infer_sample_input(loaders_txt)
    assert sample_txt.shape[0] == 1
    assert sample_txt.ndim == 2
