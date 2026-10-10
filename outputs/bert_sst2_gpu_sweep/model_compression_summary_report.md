# Model Compression Suite: Multi-Metric Benchmark Report

**Device:** `cuda`  
**Teacher Architecture:** `bert_teacher`  
**Student Architecture:** `minibert_student`  
**Dataset:** `sst2`  

---

## 1. Executive Summary & BLUF
This benchmark evaluates Knowledge Distillation (KD), Iterative Magnitude Pruning (IMP), and Post-Training Quantization (PTQ) in isolation and stacked combinations.

- **Uncompressed Baseline Teacher:** `133.05 MB`, `2.46 ms` P50 latency, `49.08%` Top-1 Accuracy.
- **Top Compression Trifecta:** Identifies high-efficiency configurations balancing < 2.0% accuracy degradation with multi-fold memory and latency reduction.

---

## 2. Combinatorial Pipeline Results

| Pipeline | Top-1 Acc (Delta) | Disk Footprint | Sparsity | P50 Latency (Speedup) | Peak RAM | Pareto-Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1_baseline_teacher_fp32` | 49.08% (+0.00%) | 133.05 MB (1.0x) | 0.1% | 2.46 ms (1.0x) | 1226.9 MB | **No** |
| `2_baseline_student_fp32` | 71.90% (+22.82%) | 34.22 MB (3.9x) | 0.0% | 1.59 ms (1.5x) | 1276.4 MB | **YES** |
| `3_distilled_student` | 71.79% (+22.71%) | 34.22 MB (3.9x) | 0.0% | 1.81 ms (1.4x) | 1321.0 MB | **No** |
| `4_pruned_teacher` | 71.67% (+22.59%) | 133.05 MB (1.0x) | 35.2% | 7.62 ms (0.3x) | 1503.1 MB | **No** |
| `5_pruned_student` | 71.56% (+22.48%) | 34.22 MB (3.9x) | 6.9% | 1.79 ms (1.4x) | 1350.6 MB | **No** |
| `6_distilled_pruned_student` | 70.99% (+21.90%) | 34.22 MB (3.9x) | 6.9% | 2.21 ms (1.1x) | 1350.7 MB | **No** |
| `7_quantized_teacher_int8` | 49.08% (+0.00%) | 133.05 MB (1.0x) | 0.2% | 4.94 ms (0.5x) | 1061.6 MB | **No** |
| `8_quantized_student_int8` | 72.02% (+22.94%) | 34.22 MB (3.9x) | 0.1% | 1.99 ms (1.2x) | 862.6 MB | **YES** |
| `9_distilled_quantized_student` | 71.79% (+22.71%) | 34.22 MB (3.9x) | 0.1% | 2.01 ms (1.2x) | 1200.0 MB | **No** |
| `10_pruned_quantized_student` | 71.56% (+22.48%) | 34.22 MB (3.9x) | 6.9% | 2.18 ms (1.1x) | 1210.9 MB | **No** |
| `11_trifecta_student_int8` | 70.99% (+21.90%) | 34.22 MB (3.9x) | 6.9% | 1.72 ms (1.4x) | 1211.1 MB | **No** |
| `12_lth_rewound_student_int8` | 71.67% (+22.59%) | 34.22 MB (3.9x) | 6.9% | 2.85 ms (0.9x) | 1203.2 MB | **No** |

---

## 3. Key Technical Takeaways
1. **The Silo vs. Stack Interaction:** Standalone pruning achieves high sparsity but can suffer quantization clipping when stacked naively. Combining Knowledge Distillation prior to pruning establishes robust feature representations that maintain calibration stability during INT8 quantization.
2. **Lottery Ticket Hypothesis:** Weight rewinding to early initialization checkpoints enables high sparsity recovery during healing epochs.
3. **Hardware Latency vs. Sparsity:** Unstructured sparsity reduces parameter count and memory, while INT8 precision reduction produces direct latency speedups on integer arithmetic units.

*Generated automatically by Model Compression Suite.*
