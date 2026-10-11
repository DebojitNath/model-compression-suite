"""
Dataset loaders and deterministic benchmark generators for vision and NLP.
Provides 3 primary benchmark datasets out of the box:
  1. synthetic (synthetic_vision / synthetic_text)
  2. cifar10 (computer vision benchmark)
  3. sst2 (Stanford Sentiment Treebank GLUE benchmark for NLP)
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import os
import re
import csv
import io
import zipfile
import binascii
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms


class SyntheticVisionDataset(Dataset):
    """
    Deterministic synthetic vision dataset producing (C, H, W) images and classification targets.
    Used for instant CI testing and offline CPU/GPU benchmarking without external downloads.
    """
    def __init__(self, num_samples: int = 1000, in_channels: int = 3, height: int = 32, width: int = 32, num_classes: int = 10, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randn(num_samples, in_channels, height, width, generator=rng)
        self.targets = torch.randint(0, num_classes, (num_samples,), generator=rng)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


class SyntheticTextDataset(Dataset):
    """
    Deterministic synthetic text token dataset producing input_ids and targets.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 32, vocab_size: int = 1000, num_classes: int = 2, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len), generator=rng)
        self.targets = torch.randint(0, num_classes, (num_samples,), generator=rng)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def tokenize_sentence(sentence: str, max_seq_len: int = 128, vocab_size: int = 30522) -> torch.Tensor:
    """
    Deterministic subword-hashed tokenizer converting text sentences into PyTorch token ID tensors.
    Uses standard BERT special tokens (101: [CLS], 102: [SEP], 0: [PAD]) and CRC32 hashing
    to map words deterministically across all operating systems without external dependencies.
    """
    words = re.findall(r"\w+|[^\w\s]", sentence.lower())
    tokens = [101]  # [CLS] token at index 0 for sequence pooling
    for w in words[: max_seq_len - 2]:
        tid = (abs(binascii.crc32(w.encode("utf-8"))) % (vocab_size - 105)) + 105
        tokens.append(tid)
    tokens.append(102)  # [SEP] token
    if len(tokens) < max_seq_len:
        tokens.extend([0] * (max_seq_len - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long)


# Embedded offline fallback sentences for SST-2 in case network is disconnected
_OFFLINE_SST2_SAMPLES = [
    ("a gorgeous , thoughtful , and utterly captivating cinematic achievement .", 1),
    ("an absolute masterpiece of contemporary storytelling and visual direction .", 1),
    ("tremendous performances and brilliant pacing throughout the entire narrative .", 1),
    ("a wonderfully charming and heartwarming experience that leaves you smiling .", 1),
    ("smart , insightful , and deeply moving from beginning to end .", 1),
    ("profoundly affecting with astonishing attention to detail and sound design .", 1),
    ("a triumphant tour de force that redefines modern filmmaking .", 1),
    ("dazzling visuals and superb acting elevate this into an instant classic .", 1),
    ("one of the most invigorating and intellectually rewarding films of the year .", 1),
    ("a joyful celebration of cinema that deserves every accolade it receives .", 1),
    ("dull , tedious , and agonizingly predictable from start to finish .", 0),
    ("an incoherent mess of cliches and uninspired cardboard characters .", 0),
    ("wastes a talented cast on an utterly shallow and meandering screenplay .", 0),
    ("painfully slow , relentlessly bleak , and completely devoid of purpose .", 0),
    ("a colossal disappointment that fails on almost every creative level .", 0),
    ("unintentionally laughable dialogue and wooden , unconvincing acting .", 0),
    ("disjointed , hollow , and thoroughly unengaging throughout its runtime .", 0),
    ("lacks any semblance of emotional depth , tension , or genuine surprise .", 0),
    ("a derivative and lifeless exercise in lazy Hollywood filmmaking .", 0),
    ("monotonous and irritating , it feels twice as long as its running time .", 0),
] * 20  # Replicated to provide sufficient calibration and evaluation batches offline


class SST2Dataset(Dataset):
    """
    Stanford Sentiment Treebank (SST-2) binary sentiment classification benchmark dataset.
    The quintessential NLP evaluation benchmark for language model distillation and compression.
    Downloads and caches GLUE SST-2 into `data_dir/SST-2/`, with automatic offline fallback.
    """
    GLUE_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

    def __init__(
        self,
        split: str = "train",
        data_dir: str = "./data",
        max_seq_len: int = 128,
        vocab_size: int = 30522,
        download: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.split = split.lower().strip()
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.data_dir = data_dir
        self.samples = []

        sst_dir = os.path.join(data_dir, "SST-2")
        target_file = os.path.join(sst_dir, "dev.tsv" if self.split in ["dev", "val", "test"] else "train.tsv")

        if download and not os.path.exists(target_file):
            self._download_and_extract(sst_dir)

        if os.path.exists(target_file):
            self._load_tsv(target_file, max_samples)
        else:
            # Offline fallback
            for s, l in _OFFLINE_SST2_SAMPLES:
                self.samples.append((tokenize_sentence(s, self.max_seq_len, self.vocab_size), torch.tensor(l, dtype=torch.long)))

    def _download_and_extract(self, dest_dir: str):
        try:
            os.makedirs(dest_dir, exist_ok=True)
            print(f"[*] Downloading GLUE SST-2 benchmark dataset to {dest_dir}...")
            req = urllib.request.Request(self.GLUE_URL, headers={"User-Agent": "ModelCompressionSuite/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                content = resp.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(self.data_dir)
            print("[+] GLUE SST-2 downloaded and cached successfully.")
        except Exception as e:
            print(f"[!] Note: Could not download SST-2 over network ({e}). Using built-in offline dataset.")

    def _load_tsv(self, file_path: str, max_samples: Optional[int] = None):
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    sentence, label_str = row[0], row[1]
                    try:
                        label = int(label_str)
                    except ValueError:
                        continue
                    tokens = tokenize_sentence(sentence, self.max_seq_len, self.vocab_size)
                    self.samples.append((tokens, torch.tensor(label, dtype=torch.long)))
                    if max_samples and len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def get_dataloaders(
    dataset_name: str = "synthetic_vision",
    batch_size: int = 64,
    data_dir: str = "./data",
    num_classes: int = 10,
    calib_samples: int = 256,
    num_workers: int = 0,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Factory function returning train, val, test, and PTQ calibration dataloaders.

    Args:
        dataset_name: "synthetic", "synthetic_vision", "synthetic_text", "cifar10", "sst2", or directory path
        batch_size: Batch size for training and testing
        data_dir: Path to directory for downloading datasets
        num_classes: Number of classification targets
        calib_samples: Number of samples dedicated to PTQ activation calibration
        num_workers: DataLoader worker count
        seed: Random seed for train/val split
        max_samples: Optional maximum number of samples to load (useful for fast CPU prototyping)

    Returns:
        Dictionary with keys: "train", "val", "test", "calib"
    """
    name = dataset_name.lower().strip()

    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_full = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

        # 45,000 train / 5,000 val
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(train_full, [45000, 5000], generator=generator)

        if max_samples and max_samples < len(train_set):
            train_set = Subset(train_set, list(range(max_samples)))
            val_set = Subset(val_set, list(range(min(len(val_set), max(100, max_samples // 5)))))
            test_set = Subset(test_set, list(range(min(len(test_set), max(200, max_samples // 4)))))

        # Calibration subset
        calib_indices = list(range(min(calib_samples, len(train_set))))
        calib_set = Subset(train_set, calib_indices)

    elif name in ["sst2", "sst-2", "glue_sst2"]:
        train_full = SST2Dataset(split="train", data_dir=data_dir, max_seq_len=128, vocab_size=30522, download=True, max_samples=max_samples)
        test_set = SST2Dataset(split="test", data_dir=data_dir, max_seq_len=128, vocab_size=30522, download=True, max_samples=max_samples)

        n_val = max(1, min(1000, int(len(train_full) * 0.1)))
        n_train = len(train_full) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

        calib_indices = list(range(min(calib_samples, len(train_set))))
        calib_set = Subset(train_set, calib_indices)

    elif name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_full = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(train_full, [45000, 5000], generator=generator)
        calib_indices = list(range(min(calib_samples, len(train_set))))
        calib_set = Subset(train_set, calib_indices)

    elif os.path.isdir(dataset_name) or (data_dir and os.path.isdir(data_dir) and name in ["image_folder", "custom", "custom_folder"]):
        target_dir = dataset_name if os.path.isdir(dataset_name) else data_dir
        # Standard generic image transforms
        transform_train = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
        if "train" in subdirs and "test" in subdirs:
            train_full = torchvision.datasets.ImageFolder(os.path.join(target_dir, "train"), transform=transform_train)
            test_set = torchvision.datasets.ImageFolder(os.path.join(target_dir, "test"), transform=transform_test)
            if "val" in subdirs:
                val_set = torchvision.datasets.ImageFolder(os.path.join(target_dir, "val"), transform=transform_test)
                train_set = train_full
            else:
                n_val = max(1, int(len(train_full) * 0.15))
                n_train = len(train_full) - n_val
                train_set, val_set = random_split(train_full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
        elif subdirs:
            # Class subdirectories directly inside target_dir
            full_set = torchvision.datasets.ImageFolder(target_dir, transform=transform_train)
            total = len(full_set)
            n_test = max(1, int(total * 0.15))
            n_val = max(1, int(total * 0.15))
            n_train = total - n_test - n_val
            train_set, val_set, test_set = random_split(
                full_set, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
            )
        else:
            raise ValueError(
                f"Dataset directory '{target_dir}' does not contain any class subfolders for ImageFolder.\n"
                f"Expected class subdirectories (e.g. {target_dir}/class1/, {target_dir}/class2/) "
                f"or standard splits ({target_dir}/train/, {target_dir}/test/)."
            )

        calib_indices = list(range(min(calib_samples, len(train_set))))
        calib_set = Subset(train_set, calib_indices)

    elif name == "synthetic_text":
        full_set = SyntheticTextDataset(num_samples=1200, seq_len=32, num_classes=num_classes, seed=seed)
        train_set, val_set, test_set = random_split(full_set, [800, 200, 200], generator=torch.Generator().manual_seed(seed))
        calib_indices = list(range(min(calib_samples, len(train_set))))
        calib_set = Subset(train_set, calib_indices)

    else:  # default "synthetic" or "synthetic_vision"
        full_set = SyntheticVisionDataset(num_samples=1200, num_classes=num_classes, seed=seed)
        train_set, val_set, test_set = random_split(full_set, [800, 200, 200], generator=torch.Generator().manual_seed(seed))
        calib_indices = list(range(min(calib_samples, len(train_set))))
        calib_set = Subset(train_set, calib_indices)

    dataloaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "calib": DataLoader(calib_set, batch_size=min(32, calib_samples), shuffle=False, num_workers=num_workers),
    }
    return dataloaders


def infer_sample_input(dataloaders: Dict[str, DataLoader], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Extracts a batch-1 sample input tensor from the test loader for latency profiling.
    """
    device = device or torch.device("cpu")
    for batch in dataloaders["test"]:
        item = batch[0] if isinstance(batch, (list, tuple)) else batch
        return item[:1].to(device)
    raise RuntimeError("Test dataloader is empty; cannot infer sample input.")
