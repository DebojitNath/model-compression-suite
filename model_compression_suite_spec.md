# MASTER PROJECT CONTEXT SPECIFICATION
**Project Name:** Model Compression Suite (Knowledge Distillation, Iterative Pruning & Post-Training Quantization)  
**Document Version:** 1.0.0  
**Target Audience:** Autonomous AI Agents, Technical Implementers, Systems Reviewers  

---

## 1. BLUF (Bottom Line Up Front)
The Model Compression Suite is an end-to-end PyTorch benchmarking framework engineered to systematically apply and evaluate Knowledge Distillation, Iterative Magnitude Pruning, and Post-Training Quantization in combination rather than in isolation. Its ultimate goal is to generate empirically verified Pareto frontiers mapping the trade-offs between task accuracy, parameter count, memory footprint, and inference latency on resource-constrained deployment environments.

---

## 2. THE 5W2H TECHNICAL SPECIFICATION

### WHAT (System Architecture & Component Definition)
A modular, pure-Python optimization framework built on PyTorch that processes trained baseline models through three sequential/combinatorial compression pipelines and profiles the resultant artifacts:
* **Knowledge Distillation Module:** Teacher-Student training loop utilizing a composite loss function:
  $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}}(y, \hat{y}_{\text{student}}) + (1 - \alpha) T^2 \mathcal{L}_{\text{KL}}\left(\text{Softmax}\left(\frac{z_{\text{student}}}{T}\right), \text{Softmax}\left(\frac{z_{\text{teacher}}}{T}\right)\right)$$
  where $\alpha$ is a loss weighting coefficient, $T$ is the temperature hyperparameter, and $z$ represents unnormalized logits.
* **Iterative Magnitude Pruner (IMP):** Custom parameter masking engine that calculates weight tensor absolute magnitudes, applies binary masks to zero out the bottom $P\%$ of weights, and executes scheduled fine-tuning/healing epochs per iteration (incorporating Lottery Ticket Hypothesis weight rewinding principles).
* **Post-Training Quantization (PTQ) & Calibration Engine:** Precision reduction pipeline (FP32 $\to$ FP16 / INT8) implementing symmetric and asymmetric mapping:
  $$q = \text{clip}\left(\text{round}\left(\frac{x}{s}\right) + z, q_{\text{min}}, q_{\text{max}}\right)$$
  including dynamic/static calibration passes to determine optimal scale ($s$) and zero-point ($z$) without catastrophic dynamic range clipping.
* **Pareto Profiling & Visualization Suite:** Automated harness that measures top-line task accuracy, model artifact disk size (MB), peak VRAM/RAM allocation (MB), and execution latency (P50/P90/P99 ms), plotting 2D/3D Pareto-optimal frontiers.

### WHO (Target Audience & Stakeholders)
* **Primary User:** ML Engineers, ML Infrastructure Engineers, and Edge AI Developers seeking mathematically grounded compression strategies for edge or CPU-cloud deployments.
* **Evaluation Stakeholders:** Technical hiring managers, senior ML systems interviewers, and applied researchers evaluating candidate depth in PyTorch internals, numerical precision, and hardware-conscious ML.

### WHY (Problem Statement & Value Proposition)
* **The Industry Gap:** Modern deep learning architectures are computationally prohibitive to serve at scale. While research papers present distillation, pruning, and quantization in silos, stacking them in production often triggers unpredictable accuracy collapse due to non-linear interaction (e.g., extreme sparsity distorting quantization calibration ranges).
* **The Solution:** This project replaces guesswork with a unified experimentation harness that reveals exactly how much compression can be extracted before hitting diminishing returns or catastrophic failure.

### WHERE (Deployment & Execution Scope)
* **Development / Optimization Environment:** Local machine or cloud VM equipped with a single consumer/cloud NVIDIA GPU (or standard multi-core CPU).
* **Target Deployment Profile:** Low-power CPU instances (e.g., AWS EC2 `t3`/`t4g`), embedded systems, or edge devices where GPU acceleration is unavailable or cost-prohibitive.

### WHEN (Operational Lifecycle)
* **Execution Model:** Offline, post-training optimization lifecycle (Batch execution).
* **Runtime Triggers:** Triggered post-baseline model convergence to compress, calibrate, benchmark, and export production-ready inference artifacts prior to deployment packaging.

### HOW (Technology Stack & Implementation Constraints)
* **Language:** Pure Python (3.10+).
* **Core Libraries:** `torch`, `torch.nn`, `torch.nn.functional` (No high-level black-box wrappers).
* **Numerical & Profiling Tools:** `numpy`, `scipy`, `matplotlib`, `seaborn`, `torch.profiler`, `time`.
* **Target Baseline Models:** Standard benchmark architectures (e.g., BERT-base distilled to a 4-layer MiniBERT/TinyBERT for NLP, or ResNet-50 distilled to ResNet-18 for Computer Vision).
* **Datasets:** Standard open-source benchmarks (e.g., GLUE / SST-2 for text classification; CIFAR-10 / CIFAR-100 for vision).

### HOW MUCH / SCALE (Performance Constraints & Targets)
* **Model Scale:** Architectures ranging from 10M to 110M parameters.
* **Target Compression Thresholds:**
  * Model size reduction: **3x to 5x** decrease in disk footprint and memory footprint.
  * Latency speedup: **2x to 3.5x** reduction in CPU batch-1 inference time.
  * Accuracy tolerance: Maximum accuracy degradation $\le \mathbf{1.5\% - 2.0\%}$ relative to the uncompressed FP32 teacher baseline.
* **Evaluation Rigor:** Benchmarking sweeps across $\ge 10$ distinct pipeline combinations (e.g., Baseline FP32 $\to$ Distilled Student $\to$ Pruned Student $\to$ Quantized Student $\to$ Distilled+Pruned+Quantized).

---

## 3. HIGH-IMPACT RESUME BULLET POINTS

* **Architected an end-to-end model compression suite in PyTorch**, reducing transformer disk and memory footprint by **[X%]** (from **[X MB]** to **[X MB]**) and cutting CPU inference latency by **[X%]** with less than **[X%]** accuracy degradation across benchmark datasets by systematically coupling Knowledge Distillation, Iterative Magnitude Pruning, and Post-Training Quantization.
* **Engineered an Iterative Magnitude Pruning (IMP) engine** with custom dynamic tensor masks and weight rewinding schedules, achieving **[X%]** parameter sparsity while recovering **[X%]** of initial accuracy drop through structured retraining cycles.
* **Implemented a custom Post-Training Quantization (PTQ) pipeline (FP32 to INT8/FP16)**, developing statistical calibration routines (Min-Max and Percentile) to compute optimal scaling factors and zero-points, mitigating quantization noise and reducing peak VRAM allocation by **[X%]**.
* **Built an automated Pareto-frontier benchmarking and visualization harness**, profiling multi-dimensional trade-offs across parameter counts, P50/P99 latencies, and accuracy curves across **[X]** distinct compression configurations to establish mathematically optimal deployment profiles.

---

## 4. ESTABLISHED SCOPE BOUNDARIES (Anti-Hallucination Guardrails)

To maintain absolute implementation fidelity, future agents and implementers must strictly adhere to these explicit constraints:

### Explicitly OUT OF SCOPE (Do NOT Implement):
1. **No Custom C++/CUDA Kernels:** All tensor manipulation, masking, distillation loss calculation, and calibration must be executed purely via Python and standard PyTorch APIs. Do not introduce C++, CUDA extensions, or Triton kernels.
2. **No Heavy Serving Engines:** Do not integrate TensorRT, TensorRT-LLM, vLLM, ONNX Runtime, or Triton Inference Server. The goal is to build the *compression and evaluation engine*, not a distributed inference server.
3. **No Massive 70B+ LLM Distributed Training:** Do not attempt multi-node or distributed tensor-parallel training across massive LLMs. Keep the model scope strictly between 10M and 110M parameters to ensure reproducibility on single-GPU/CPU setups.
4. **No Web UI / Frontend Dashboard:** Do not build React, Vue, Streamlit, or Gradio interfaces. Output artifacts must consist strictly of modular Python packages, CLI entry points, exported PyTorch model weights (`.pt`), structured logs (JSON/CSV), and saved Matplotlib/Seaborn Pareto graphs (`.png`/`.svg`).
5. **No Quantization-Aware Training (QAT):** Scope is strictly constrained to Post-Training Quantization (PTQ) applied after Distillation and Iterative Pruning, avoiding the added architectural complexity of simulated fake-quantization backward graphs during training.
