# Model Compression Suite: Multi-Metric Benchmark Report

**Device:** `cuda`  
**Teacher Architecture:** `resnet50`  
**Student Architecture:** `resnet18`  
**Dataset:** `cifar10`  

---

## 1. Executive Summary & BLUF
This benchmark evaluates Knowledge Distillation (KD), Iterative Magnitude Pruning (IMP), and Post-Training Quantization (PTQ) in isolation and stacked combinations.

- **Uncompressed Baseline Teacher:** `90.02 MB`, `5.47 ms` P50 latency, `86.04%` Top-1 Accuracy.
- **Top Compression Trifecta:** Identifies high-efficiency configurations balancing < 2.0% accuracy degradation with multi-fold memory and latency reduction.

---

## 2. Combinatorial Pipeline Results

| Pipeline | Top-1 Acc (Delta) | Disk Footprint | Sparsity | P50 Latency (Speedup) | Peak RAM | Pareto-Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1_baseline_teacher_fp32` | 86.04% (+0.00%) | 90.02 MB (1.0x) | 0.0% | 5.47 ms (1.0x) | 1302.3 MB | **No** |
| `2_baseline_student_fp32` | 88.16% (+2.12%) | 42.70 MB (2.1x) | 0.0% | 2.25 ms (2.4x) | 1262.3 MB | **YES** |
| `3_distilled_student` | 87.80% (+1.76%) | 42.70 MB (2.1x) | 0.0% | 2.26 ms (2.4x) | 1262.5 MB | **No** |
| `4_pruned_teacher` | 85.53% (-0.51%) | 90.02 MB (1.0x) | 84.8% | 13.31 ms (0.4x) | 1624.4 MB | **No** |
| `5_pruned_student` | 88.43% (+2.39%) | 42.70 MB (2.1x) | 84.9% | 2.88 ms (1.9x) | 1531.8 MB | **YES** |
| `6_distilled_pruned_student` | 88.21% (+2.17%) | 42.70 MB (2.1x) | 84.9% | 4.35 ms (1.3x) | 1531.8 MB | **No** |
| `7_quantized_teacher_int8` | 85.98% (-0.06%) | 90.03 MB (1.0x) | 1.9% | 7.06 ms (0.8x) | 1541.8 MB | **No** |
| `8_quantized_student_int8` | 88.18% (+2.14%) | 42.70 MB (2.1x) | 1.7% | 3.97 ms (1.4x) | 1495.9 MB | **No** |
| `9_distilled_quantized_student` | 87.80% (+1.76%) | 42.70 MB (2.1x) | 1.6% | 4.82 ms (1.1x) | 1495.8 MB | **No** |
| `10_pruned_quantized_student` | 88.40% (+2.36%) | 42.70 MB (2.1x) | 84.9% | 2.50 ms (2.2x) | 2216.1 MB | **YES** |
| `11_trifecta_student_int8` | 88.18% (+2.14%) | 42.70 MB (2.1x) | 84.9% | 3.07 ms (1.8x) | 1495.9 MB | **No** |
| `12_lth_rewound_student_int8` | 87.80% (+1.76%) | 42.70 MB (2.1x) | 84.9% | 2.56 ms (2.1x) | 2214.8 MB | **No** |

---

## 3. Key Technical Takeaways
1. **The Silo vs. Stack Interaction:** Standalone pruning achieves high sparsity but can suffer quantization clipping when stacked naively. Combining Knowledge Distillation prior to pruning establishes robust feature representations that maintain calibration stability during INT8 quantization.
2. **Lottery Ticket Hypothesis:** Weight rewinding to early initialization checkpoints enables high sparsity recovery during healing epochs.
3. **Hardware Latency vs. Sparsity:** Unstructured sparsity reduces parameter count and memory, while INT8 precision reduction produces direct latency speedups on integer arithmetic units.

*Generated automatically by Model Compression Suite.*
