# Model Compression Suite & Pareto Profiling Engine
## Comprehensive Project Specification, Architecture & Technical Deep Dive

---

## 1. Executive Summary & Core Mission

### The Industry Challenge
Modern deep learning models (such as deep ResNets and Transformers like BERT) achieve state-of-the-art accuracy at the cost of tens to hundreds of millions of parameters, high floating-point operations (FLOPs), multi-gigabyte memory footprints, and heavy compute latency. When deploying to edge hardware, embedded devices, or low-cost cloud CPUs, these models are often computationally and economically unviable.

While the academic literature frequently investigates **Knowledge Distillation (KD)**, **Iterative Magnitude Pruning (IMP)**, and **Post-Training Quantization (PTQ)** in isolation, deploying them together in production often results in unpredictable accuracy degradation or failure. For example, aggressive parameter sparsity can distort activation distributions, leading to catastrophic quantization clipping noise during low-bit mapping.

### The Solution: Model Compression Suite
**Model Compression Suite** is a modular, production-ready, pure-PyTorch optimization and benchmarking framework. It systematically evaluates compression techniques individually and combinatorially across **12 distinct pipeline topologies**. 

Instead of relying on guesswork, the suite profiles every variant across four critical dimensions:
1. **Task Accuracy** (Top-1 Accuracy / Classification Loss)
2. **Execution Latency** (P50, P90, P99 percentile latency via GPU/CPU warmups & synchronized timings)
3. **Storage & Memory Footprint** (Disk size in MB, active parameter counts, peak RAM, peak VRAM)
4. **Parameter Sparsity** (Zero-weight ratios across convolutional and linear layers)

The end product is a set of **empirically validated 2D and 3D Pareto frontiers**, identifying the mathematical sweet spots where maximum throughput and memory reduction are achieved with negligible or zero accuracy degradation.

```
                             ┌────────────────────────────────────────┐
                             │       Trained Teacher Baseline         │
                             └───────────────────┬────────────────────┘
                                                 │
                     ┌───────────────────────────┼───────────────────────────┐
                     │                           │                           │
                     ▼                           ▼                           ▼
        [Knowledge Distillation]      [Iterative Pruning (IMP)]    [Post-Training Quantization]
         - Softmax Temperature T       - Absolute Magnitude Weight  - Min-Max & Percentile Grid
         - CE + KL Divergence Loss       Masking                    - Symmetric & Asymmetric INT8
         - Student Architecture        - Lottery Ticket Rewinding   - SQNR Validation
                     │                           │                           │
                     └───────────────────────────┼───────────────────────────┘
                                                 │
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │    12 Combinatorial Pipeline Runs      │
                             │   (Individual, Pairwise, Trifecta)     │
                             └───────────────────┬────────────────────┘
                                                 │
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │        Unified Profiler Engine         │
                             │  - P50/P90/P99 Synchronized Latency    │
                             │  - Peak RAM & VRAM Allocation          │
                             │  - Disk Size & Active Parameter Ratios │
                             └───────────────────┬────────────────────┘
                                                 │
                                                 ▼
                             ┌────────────────────────────────────────┐
                             │       Pareto Multi-Objective Grid      │
                             │  - 2D / 3D Frontier Convex Hulls       │
                             │  - Checkpoints (.pt) + manifest.json   │
                             │  - Publication-Grade Plots (PNG/SVG)   │
                             └────────────────────────────────────────┘
```

---

## 2. Theoretical & Mathematical Foundations

### A. Knowledge Distillation (KD)
Knowledge Distillation transfers dark knowledge (soft class probabilities and inter-class correlations) from a large, over-parameterized Teacher model ($T$) to a compact, compute-efficient Student model ($S$).

Given ground-truth labels $y$, teacher unnormalized logits $z_T$, and student logits $z_S$, the loss function is defined as:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}}(y, \sigma(z_S)) + (1 - \alpha) \cdot T_{\text{temp}}^2 \cdot \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_S}{T_{\text{temp}}}\right) \parallel \sigma\left(\frac{z_T}{T_{\text{temp}}}\right)\right)$$

Where:
- $\sigma(\cdot)$ is the softmax function: $\sigma(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$.
- $T_{\text{temp}} \ge 1$ is the temperature hyperparameter softening the probability distributions. As $T_{\text{temp}} \to \infty$, logits approach a uniform distribution, amplifying information in non-target logits.
- $T_{\text{temp}}^2$ scales the gradient magnitude to match the scale of standard cross-entropy gradients.
- $\alpha \in [0, 1]$ balances hard-label ground truth supervision against soft-target teacher guidance.

### B. Iterative Magnitude Pruning (IMP) & Lottery Ticket Hypothesis (LTH)
Unstructured magnitude pruning removes parameters with the smallest absolute values under the premise that near-zero weights contribute least to model output.

1. **Parameter Masking**: For each weight tensor $W \in \mathbb{R}^{d}$, a binary mask tensor $M \in \{0, 1\}^d$ is registered:
   $$\widehat{W} = W \odot M$$
   During the backward pass, gradients are masked ($\nabla_W \mathcal{L} \odot M$) to prevent zeroed weights from regrowing during optimization.

2. **Iterative Schedule**: Rather than pruning $P\%$ in one drastic step (one-shot pruning), the suite prunes iteratively over $K$ rounds with target intermediate sparsity $s_k$:
   $$s_k = P \cdot \left(1 - \left(1 - \frac{k}{K}\right)^3\right)$$
   Each round is followed by $E_{\text{healing}}$ fine-tuning epochs to let surviving parameters recover lost accuracy.

3. **Lottery Ticket Hypothesis (LTH) Weight Rewinding**:
   Frankle & Carbin (2018) demonstrated that dense networks contain sparse sub-networks ("winning lottery tickets") that can match the dense model's test accuracy when trained in isolation from their original initialization:
   $$\theta_0 \xrightarrow{\text{train}} \theta_t \xrightarrow{\text{prune}} m \odot \theta_t \xrightarrow{\text{rewind}} m \odot \theta_0 \xrightarrow{\text{retrain}} \theta^*_{\text{sparse}}$$
   The suite snapshots initial weights $\theta_0$ before training, applies the learned mask $m$, rewinds unpruned weights back to $\theta_0$, and retrains to full convergence.

### C. Post-Training Quantization (PTQ) & Calibration
Quantization maps 32-bit floating-point weights and activations ($x \in \mathbb{R}$) into 8-bit integers ($q \in [-128, 127]$ or $[0, 255]$).

1. **Uniform Affine Quantization**:
   $$q = \text{clip}\left(\left\lfloor \frac{x}{s} \right\rceil + z, q_{\min}, q_{\max}\right)$$
   $$\hat{x} = s \cdot (q - z)$$
   Where $s = \frac{\beta - \alpha}{q_{\max} - q_{\min}}$ is the scale factor, $z = \left\lfloor -\frac{\alpha}{s} \right\rceil + q_{\min}$ is the zero-point, and $[\alpha, \beta]$ defines the clipping dynamic range.

2. **Calibration Strategies**:
   - **Min-Max Calibration**: Sets $\alpha = \min(X)$ and $\beta = \max(X)$. Simple, but sensitive to outliers (a single outlier activation expands $s$, deteriorating quantization resolution across typical activations).
   - **Percentile / Histogram Calibration**: Sets $\alpha = P_{0.01}(X)$ and $\beta = P_{99.99}(X)$, clipping extreme statistical tails to preserve maximum resolution for $99.9\%$ of the distribution.

3. **Signal-to-Quantization-Noise Ratio (SQNR)**:
   The suite calculates the reconstruction fidelity of quantized tensors:
   $$\text{SQNR (dB)} = 10 \cdot \log_{10}\left(\frac{\sum x^2}{\sum (x - \hat{x})^2}\right)$$

### D. Multi-Objective Pareto Optimality
An operating point $A$ dominates operating point $B$ ($A \succ B$) if and only if:
$$\forall i \in \{1, \dots, M\}, \quad f_i(A) \ge f_i(B) \quad \text{and} \quad \exists j \in \{1, \dots, M\}, \quad f_j(A) > f_j(B)$$
where metrics to maximize include **Top-1 Accuracy**, and metrics to minimize include **Latency** and **Disk/Memory Footprint**. 

The **Pareto Frontier** is the set of all non-dominated models $\mathcal{P}^* = \{A \in \mathcal{M} \mid \nexists B \in \mathcal{M}, B \succ A\}$. A deployment engineer can select an optimal model along this frontier according to strict Service Level Agreements (SLAs).

---

## 3. The 12 Combinatorial Compression Pipelines

The benchmark engine executes a complete $2^3 + 4$ matrix representing all isolated, pairwise, and trifecta combinations:

| # | Pipeline Name | Description & Processing Order |
| :---: | :--- | :--- |
| `[1]` | `1_baseline_teacher_fp32` | Uncompressed Teacher model (e.g. ResNet-50 / BERT-base) in full FP32 precision. Serves as ground-truth anchor. |
| `[2]` | `2_baseline_student_fp32` | Uncompressed Student model (e.g. ResNet-18 / MiniBERT) trained standalone from scratch via cross-entropy. |
| `[3]` | `3_distilled_student` | Student trained via Knowledge Distillation from the trained Teacher logits ($T_{\text{temp}}=4.0, \alpha=0.5$). |
| `[4]` | `4_pruned_teacher` | Teacher model compressed via Iterative Magnitude Pruning ($85\%$ sparsity) + fine-tuning healing. |
| `[5]` | `5_pruned_student` | Baseline Student compressed via Iterative Magnitude Pruning ($85\%$ sparsity) + fine-tuning healing. |
| `[6]` | `6_distilled_pruned_student` | Student trained via KD, followed by Iterative Magnitude Pruning with KD loss during healing epochs. |
| `[7]` | `7_quantized_teacher_int8` | Teacher model quantized to INT8 precision using percentile activation calibration. |
| `[8]` | `8_quantized_student_int8` | Standalone Student quantized to INT8 precision using percentile activation calibration. |
| `[9]` | `9_distilled_quantized_student` | Distilled Student directly converted to INT8 precision via PTQ. |
| `[10]` | `10_pruned_quantized_student` | Pruned Baseline Student converted to INT8 precision (evaluates sparsity + PTQ interaction). |
| `[11]` | `11_trifecta_student_int8` | Full Trifecta: Student is Distilled, then Pruned via IMP, then Quantized to INT8. |
| `[12]` | `12_lth_rewound_student_int8` | Advanced Trifecta: Student is Distilled, Pruned with LTH weight rewinding to initial checkpoint, then Quantized to INT8. |

---

## 4. Software Architecture & Codebase Layout

The repository is structured as a standalone, modular Python library with an extensible API and CLI interface:

```
model-compression-suite/
├── compression_suite/              # Core Library Package
│   ├── __init__.py                 # Package version and export definitions
│   ├── cli.py                      # Production CLI entrypoint (sweep, interactive, list-models)
│   │
│   ├── distillation/               # Knowledge Distillation Subsystem
│   │   ├── __init__.py
│   │   ├── loss.py                 # DistillationLoss (CE + KL Divergence with temperature)
│   │   └── trainer.py              # DistillationTrainer (epoch loop, scheduler, evaluation)
│   │
│   ├── pruning/                    # Pruning & Sparsity Subsystem
│   │   ├── __init__.py
│   │   ├── mask.py                 # MaskManager (global absolute magnitude thresholding)
│   │   └── pruner.py               # IterativeMagnitudePruner (IMP, LTH rewinding, auto-sparsity grid)
│   │
│   ├── quantization/               # Post-Training Quantization Subsystem
│   │   ├── __init__.py
│   │   ├── calibrator.py           # ActivationCalibrator (Min-Max & Percentile statistics)
│   │   ├── quantizer.py            # PostTrainingQuantizer (INT8 scale/zero-point mapper)
│   │   └── simulated.py            # SimulatedPTQModel (CUDA/CPU simulated fake-quantized inference)
│   │
│   ├── models/                     # Model Zoo & Registry Subsystem
│   │   ├── __init__.py
│   │   ├── registry.py             # ModelRegistry with dynamic factory decorator
│   │   ├── fast_convnet.py         # Lightweight 4-layer and 6-layer CNN architectures
│   │   ├── resnet.py               # ResNet-18, ResNet-34, and ResNet-50 implementations
│   │   ├── bert.py                 # Transformer classification (BERT-base & 4-layer MiniBERT)
│   │   └── loader.py               # Checkpoint loader for custom weights and bundle directories
│   │
│   ├── data/                       # Datasets & Tokenization Subsystem
│   │   ├── __init__.py
│   │   ├── datasets.py             # get_dataloaders factory & automatic sample input derivation
│   │   ├── cifar.py                # CIFAR-10 and CIFAR-100 download & augmentations
│   │   ├── sst2.py                 # Stanford Sentiment Treebank (SST-2) with WordPiece tokenizer
│   │   └── synthetic.py            # Synthetic vision & sequence generators for instant CI/CD testing
│   │
│   ├── profiling/                  # Unified Benchmarking Engine
│   │   ├── __init__.py
│   │   ├── latency.py              # BenchmarkTimer (CUDA sync, warmup cycles, P50/P90/P99)
│   │   ├── memory.py               # MemoryTracker (psutil RAM, torch.cuda peak VRAM, parameter count)
│   │   ├── pareto.py               # ParetoFrontierCalculator (2D & 3D non-dominated filtering)
│   │   └── unified.py              # UnifiedProfiler (orchestrates latency, memory, size, accuracy)
│   │
│   ├── visualization/              # Publication Visuals Subsystem
│   │   ├── __init__.py
│   │   └── plots.py                # 2D Pareto, 3D Convex Hull, Pipeline Bars, Sparsity Stress Curves
│   │
│   ├── pipeline/                   # Sweep Orchestration & Staging
│   │   ├── __init__.py
│   │   ├── runner.py               # PipelineRunner (12-pipeline coordinator + fault-tolerant staging)
│   │   └── experiment.py           # Experiment persistence (JSON metrics, CSV, Markdown reports)
│   │
│   └── utils/                      # Helper Utilities
│       ├── __init__.py
│       └── workspace.py            # Directory resolution, bundle validation, collision handling
│
├── configs/                        # Example YAML Configuration Files
│   ├── resnet_cifar10_sweep.yaml   # Vision sweep configuration
│   └── bert_sst2_sweep.yaml        # NLP transformer configuration
│
├── tests/                          # 28 Comprehensive Pytest Suites
│   ├── test_models.py              # Model registration, architecture parameter checks
│   ├── test_datasets.py            # Vision, NLP, and synthetic data loading
│   ├── test_distillation.py        # KD loss mathematical invariance & training step
│   ├── test_pruning.py             # Masking, gradient zeroing, LTH rewinding, auto-sparsity
│   ├── test_quantization.py        # Calibration clipping, INT8 roundtrip SQNR
│   ├── test_profiling.py           # P50/P90/P99 latency calculations, peak memory tracking
│   ├── test_pareto.py              # 2D & 3D Pareto frontier correctness
│   ├── test_visualization.py       # Plot generator output integrity
│   ├── test_bundle_input.py        # External directory bundle ingestion
│   └── test_pipeline_integration.py# End-to-end 12-pipeline integration run
│
├── .github/workflows/ci.yml        # Multi-platform CI (Python 3.10 & 3.11 pytest verification)
├── outputs/                        # Staged output folders containing plots, reports, CSV, JSON
├── pyproject.toml                  # Modern PEP 517/621 build configuration
├── requirements.txt                # Pinned production dependencies
├── LICENSE                         # MIT Open Source License
└── README.md                       # GitHub documentation with badges, tables, and quickstarts
```

---

## 5. Fault-Tolerant Checkpoint Staging & Auto-Resume

Training 12 model pipelines (including Teacher pre-training, Student training, Distillation epochs, and multiple IMP healing cycles) requires substantial GPU compute. To protect against hardware power cuts, session interruptions, or thermal throttling, the suite incorporates a built-in **Checkpoint Staging Engine** (`.staging/`):

1. **Atomic Staging**: Immediately after each pipeline completes its training or optimization step, its full model state is persisted to disk:
   - `1_baseline_teacher_fp32.pt`
   - `2_baseline_student_fp32.pt`
   - `3_distilled_student.pt`
   - `4_pruned_teacher.pt`
   - `5_pruned_student.pt`
   - `6_distilled_pruned_student.pt`
2. **Instant Hot-Resume**: If a sweep is re-launched with the same output directory, `PipelineRunner` scans the staging directory:
   ```
   >>> [1/12] Loading staged Baseline Teacher FP32...
   >>> [2/12] Loading staged Baseline Student FP32...
   >>> [3/12] Loading staged Distilled Student...
   ```
   It skips re-training and resumes profiling and subsequent compression steps immediately.
3. **Interactive Model Export**: Upon completing the sweep, the user can selectively export models:
   - Specific IDs (e.g., `3, 11`),
   - Semantic tokens (`pareto`, `top-acc`, `top-speed`, `unpruned`, `all`),
   - Checkpoints are copied to `outputs/<sweep>/checkpoints/` alongside an auto-generated `manifest.json`.

---

## 6. Real Empirical Benchmarks

### 1. Vision Benchmark: ResNet-50 $\to$ ResNet-18 on CIFAR-10
*Hardware: NVIDIA RTX GPU | Teacher: ResNet-50 (23.5M params) | Student: ResNet-18 (11.2M params)*

| # | Pipeline Name | Top-1 Acc | P50 Latency (ms) | Speedup | Disk Size | Sparsity | Pareto? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `1_baseline_teacher_fp32` | 86.0% | 5.47 ms | 1.0x | 90.02 MB | 0.0% | - |
| `2` | `2_baseline_student_fp32` | 88.2% | 2.25 ms | 2.4x | 42.70 MB | 0.0% | **YES** |
| `3` | `3_distilled_student` | 87.8% | 2.26 ms | 2.4x | 42.70 MB | 0.0% | - |
| `4` | `4_pruned_teacher` | 85.5% | 13.31 ms | 0.4x | 90.02 MB | 84.8% | - |
| `5` | `5_pruned_student` | 88.4% | 2.88 ms | 1.9x | 42.70 MB | 84.9% | **YES** |
| `6` | `6_distilled_pruned_student` | 88.2% | 4.35 ms | 1.3x | 42.70 MB | 84.9% | - |
| `7` | `7_quantized_teacher_int8` | 86.0% | 7.06 ms | 0.8x | 90.03 MB | 1.9% | - |
| `8` | `8_quantized_student_int8` | 88.2% | 3.97 ms | 1.4x | 42.70 MB | 1.7% | - |
| `9` | `9_distilled_quantized_student` | 87.8% | 4.82 ms | 1.1x | 42.70 MB | 1.6% | - |
| `10` | `10_pruned_quantized_student` | 88.4% | 2.50 ms | 2.2x | 42.70 MB | 84.9% | **YES** |
| `11` | **`11_trifecta_student_int8`** | **88.2%** | **3.07 ms** | **1.8x** | **42.70 MB** | **84.9%** | - |
| `12` | `12_lth_rewound_student_int8` | 87.8% | 2.56 ms | 2.1x | 42.70 MB | 84.9% | - |

> **Key Architectural Insight:**
> - ResNet-18 achieves higher accuracy on CIFAR-10 (88.2% vs 86.0%) than ResNet-50 because smaller architectures resist overfitting on 32×32 resolution datasets.
> - **`[10]` and `[11]`** achieve **1.8x–2.2x lower latency**, **53% smaller model size**, and **84.9% parameter sparsity** while exceeding teacher accuracy.

---

### 2. NLP Benchmark: BERT-base $\to$ 4-Layer MiniBERT on SST-2
*Hardware: NVIDIA RTX GPU | Teacher: BERT-base (109.5M params) | Student: MiniBERT (14.4M params)*

| # | Pipeline Name | Top-1 Acc | P50 Latency (ms) | Speedup | Disk Size | Linear Sparsity | Pareto? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1` | `1_baseline_teacher_fp32` | 74.2% | 8.84 ms | 1.0x | 133.05 MB | 0.0% | - |
| `2` | `2_baseline_student_fp32` | 71.9% | 2.15 ms | 4.1x | 34.22 MB | 0.0% | **YES** |
| `3` | `3_distilled_student` | 72.8% | 2.16 ms | 4.1x | 34.22 MB | 0.0% | **YES** |
| `11` | **`11_trifecta_student_int8`** | **71.0%** | **2.32 ms** | **3.8x** | **34.22 MB** | **85.0%** | **YES** |

> **Key Architectural Insight:**
> Distillation recovers **+0.9% accuracy** over training the student from scratch, and the Trifecta compressed model delivers a **3.8x latency speedup** and **3.9x memory shrink** with less than 1% degradation from the standalone student.

---

## 7. Execution Modes & CLI Guide

The tool supports three primary modes of operation:

### Mode 1: Research Sweep (Command-Line Flags)
Trains or imports models via built-in registries and sweeps all 12 pipelines:
```powershell
python -m compression_suite.cli sweep `
  --teacher resnet50 `
  --student resnet18 `
  --dataset cifar10 `
  --epochs 2 `
  --prune-sparsity auto `
  --sparsity-candidates 0.3 0.5 0.7 0.85 `
  --output-dir ./outputs/resnet_cifar10_sweep `
  --device cuda
```

### Mode 2: Input Bundle Mode (Pre-trained Production Models)
Accepts an existing directory bundle containing pre-trained weights and test data:
```
inputs/production_bundle/
├── teacher/
│   └── model.pt (or checkpoint.pt)
├── student/
│   └── model.pt
└── dataset/
    ├── train/
    └── val/
```
Execution:
```powershell
python -m compression_suite.cli sweep `
  --input ./inputs/production_bundle `
  --output-dir ./outputs/production_eval `
  --export pareto
```

### Mode 3: Interactive Terminal Setup
Launches an interactive prompt in the terminal to configure and start a compression sweep:
```powershell
python -m compression_suite.cli interactive
```

---

## 8. Summary of Generated Output Artifacts

At the conclusion of any sweep, the output directory contains a self-contained, publication-ready evaluation report:

| File Name | Description |
| :--- | :--- |
| `results.csv` | Full metric table (Accuracy, Loss, Size, Active Params, Sparsity, P50/P90/P99 Latency, RAM, VRAM, Pareto flags). |
| `results.json` | Complete machine-readable configuration, hyperparameter metadata, and metric dictionaries. |
| `model_compression_summary_report.md` | Auto-generated Markdown report comparing baseline vs compressed models and highlighting Pareto winners. |
| `pareto_accuracy_vs_latency.png` (`.svg`) | 2D Pareto frontier curve: Top-1 Accuracy (%) vs P50 Latency (ms). |
| `pareto_accuracy_vs_size.png` (`.svg`) | 2D Pareto frontier curve: Top-1 Accuracy (%) vs Model Disk Size (MB). |
| `pareto_3d_frontier.png` (`.svg`) | Interactive 3D scatter and surface: Accuracy vs Latency vs Model Size. |
| `pipeline_comparison_bars.png` (`.svg`) | Grouped bar chart illustrating the evolution of all 12 pipelines side-by-side. |
| `sparsity_stress_curve.png` (`.svg`) | Auto-sparsity sensitivity curve displaying accuracy drop across candidate sparsity levels. |
| `checkpoints/` | Directory containing exported `.pt` weights and `manifest.json`. |

---

## 9. Technology Stack & Technical Constraints

- **Language:** Python 3.10 / 3.11
- **Core Framework:** PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
- **Scientific Computing:** `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`
- **System Telemetry:** `psutil`, `torch.cuda`
- **Configuration:** `pyyaml`
- **Testing:** `pytest` (28 unit and integration tests passing)
- **CI/CD:** GitHub Actions workflow executing on Ubuntu with matrix testing
