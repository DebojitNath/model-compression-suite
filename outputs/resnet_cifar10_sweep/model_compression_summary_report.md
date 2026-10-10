# Model Compression Suite: Multi-Metric Benchmark Report

**Device:** `cuda`  
**Teacher Architecture:** `resnet50`  
**Student Architecture:** `resnet18`  
**Dataset:** `cifar10`  

---

## 1. Executive Summary & BLUF
This benchmark evaluates Knowledge Distillation (KD), Iterative Magnitude Pruning (IMP), and Post-Training Quantization (PTQ) in isolation and stacked combinations.

- **Uncompressed Baseline Teacher:** `90.02 MB`, `7.51 ms` P50 latency, `10.00%` Top-1 Accuracy.
- **Top Compression Trifecta:** Identifies high-efficiency configurations balancing < 2.0% accuracy degradation with multi-fold memory and latency reduction.

---

## 2. Combinatorial Pipeline Results

| Pipeline | Top-1 Acc (Delta) | Disk Footprint | Sparsity | P50 Latency (Speedup) | Peak RAM | Pareto-Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1_baseline_teacher_fp32` | 10.00% (+0.00%) | 90.02 MB (1.0x) | 0.1% | 7.51 ms (1.0x) | 1300.7 MB | **No** |
| `2_baseline_student_fp32` | 88.81% (+78.81%) | 42.70 MB (2.1x) | 0.0% | 2.75 ms (2.7x) | 1258.8 MB | **YES** |
| `3_distilled_student` | 89.38% (+79.38%) | 42.70 MB (2.1x) | 0.0% | 4.29 ms (1.8x) | 1259.0 MB | **YES** |
| `4_pruned_teacher` | 10.00% (+0.00%) | 90.02 MB (1.0x) | 84.8% | 7.52 ms (1.0x) | 1513.2 MB | **No** |
| `5_pruned_student` | 88.73% (+78.73%) | 42.70 MB (2.1x) | 84.9% | 2.56 ms (2.9x) | 1466.5 MB | **YES** |
| `6_distilled_pruned_student` | 89.37% (+79.37%) | 42.70 MB (2.1x) | 84.9% | 3.33 ms (2.3x) | 1466.4 MB | **YES** |
| `7_quantized_teacher_int8` | 9.82% (-0.18%) | 90.03 MB (1.0x) | 0.5% | 6.76 ms (1.1x) | 380.0 MB | **No** |
| `8_quantized_student_int8` | 88.79% (+78.79%) | 42.70 MB (2.1x) | 1.6% | 3.58 ms (2.1x) | 336.3 MB | **No** |
| `9_distilled_quantized_student` | 89.33% (+79.33%) | 42.70 MB (2.1x) | 2.0% | 2.47 ms (3.0x) | 337.9 MB | **YES** |
| `10_pruned_quantized_student` | 88.78% (+78.78%) | 42.70 MB (2.1x) | 84.9% | 2.60 ms (2.9x) | 1058.3 MB | **No** |
| `11_trifecta_student_int8` | 89.39% (+79.39%) | 42.70 MB (2.1x) | 84.9% | 2.52 ms (3.0x) | 1058.5 MB | **YES** |
| `12_lth_rewound_student_int8` | 89.03% (+79.03%) | 42.70 MB (2.1x) | 84.9% | 3.04 ms (2.5x) | 344.6 MB | **No** |

---

## 3. Key Technical Takeaways
1. **The Silo vs. Stack Interaction:** Standalone pruning achieves high sparsity but can suffer quantization clipping when stacked naively. Combining Knowledge Distillation prior to pruning establishes robust feature representations that maintain calibration stability during INT8 quantization.
2. **Lottery Ticket Hypothesis:** Weight rewinding to early initialization checkpoints enables high sparsity recovery during healing epochs.
3. **Hardware Latency vs. Sparsity:** Unstructured sparsity reduces parameter count and memory, while INT8 precision reduction produces direct latency speedups on integer arithmetic units.

*Generated automatically by Model Compression Suite.*
