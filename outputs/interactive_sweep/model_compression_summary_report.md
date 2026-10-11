# Model Compression Suite: Multi-Metric Benchmark Report

**Device:** `cuda`  
**Teacher Architecture:** `fast_convnet_teacher`  
**Student Architecture:** `fast_convnet_student`  
**Dataset:** `synthetic`  

---

## 1. Executive Summary & BLUF
This benchmark evaluates Knowledge Distillation (KD), Iterative Magnitude Pruning (IMP), and Post-Training Quantization (PTQ) in isolation and stacked combinations.

- **Uncompressed Baseline Teacher:** `8.70 MB`, `0.64 ms` P50 latency, `7.50%` Top-1 Accuracy.
- **Top Compression Trifecta:** Identifies high-efficiency configurations balancing < 2.0% accuracy degradation with multi-fold memory and latency reduction.

---

## 2. Combinatorial Pipeline Results

| Pipeline | Top-1 Acc (Delta) | Disk Footprint | Sparsity | P50 Latency (Speedup) | Peak RAM | Pareto-Optimal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `1_baseline_teacher_fp32` | 7.50% (+0.00%) | 8.70 MB (1.0x) | 0.0% | 0.64 ms (1.0x) | 1180.3 MB | **No** |
| `2_baseline_student_fp32` | 8.50% (+1.00%) | 0.40 MB (21.8x) | 0.0% | 0.47 ms (1.4x) | 1155.0 MB | **YES** |
| `3_distilled_student` | 7.50% (+0.00%) | 0.40 MB (21.8x) | 0.0% | 0.46 ms (1.4x) | 1198.9 MB | **YES** |
| `4_pruned_teacher` | 10.00% (+2.50%) | 8.70 MB (1.0x) | 69.9% | 0.64 ms (1.0x) | 1259.2 MB | **No** |
| `5_pruned_student` | 10.50% (+3.00%) | 0.40 MB (21.8x) | 69.6% | 0.49 ms (1.3x) | 1235.7 MB | **YES** |
| `6_distilled_pruned_student` | 7.50% (+0.00%) | 0.40 MB (21.8x) | 69.6% | 0.47 ms (1.4x) | 1230.7 MB | **No** |
| `7_quantized_teacher_int8` | 7.50% (+0.00%) | 8.70 MB (1.0x) | 0.6% | 0.73 ms (0.9x) | 2002.3 MB | **No** |
| `8_quantized_student_int8` | 8.50% (+1.00%) | 0.40 MB (21.8x) | 0.6% | 0.53 ms (1.2x) | 2046.4 MB | **No** |
| `9_distilled_quantized_student` | 7.50% (+0.00%) | 0.40 MB (21.8x) | 0.6% | 0.55 ms (1.2x) | 2046.4 MB | **No** |
| `10_pruned_quantized_student` | 10.50% (+3.00%) | 0.40 MB (21.8x) | 69.6% | 0.52 ms (1.3x) | 2047.4 MB | **No** |
| `11_trifecta_student_int8` | 8.00% (+0.50%) | 0.40 MB (21.8x) | 69.6% | 0.51 ms (1.3x) | 2047.4 MB | **No** |
| `12_lth_rewound_student_int8` | 7.50% (+0.00%) | 0.40 MB (21.8x) | 69.6% | 0.50 ms (1.3x) | 2047.4 MB | **No** |

---

## 3. Key Technical Takeaways
1. **The Silo vs. Stack Interaction:** Standalone pruning achieves high sparsity but can suffer quantization clipping when stacked naively. Combining Knowledge Distillation prior to pruning establishes robust feature representations that maintain calibration stability during INT8 quantization.
2. **Lottery Ticket Hypothesis:** Weight rewinding to early initialization checkpoints enables high sparsity recovery during healing epochs.
3. **Hardware Latency vs. Sparsity:** Unstructured sparsity reduces parameter count and memory, while INT8 precision reduction produces direct latency speedups on integer arithmetic units.

*Generated automatically by Model Compression Suite.*
