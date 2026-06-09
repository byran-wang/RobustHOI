# Ablation Study

Baseline: `output[4_25_11_09][879f][sam3d_auto_selection]` (full pipeline)

All deltas = ablation − baseline. Negative = regression.

---

## Summary (Average ADD AUC)

| Variant               | Avg ADD AUC | Δ vs Baseline |
|-----------------------|-------------|---------------|
| **Baseline (full)**   | **88.10**   | —             |
| SAM3D+FP+BA           | 85.18       | -2.92         |
| no_depth_filter       | 84.49       | -3.61         |
| no_rectified_by_hand  | 82.79       | -5.31         |
| PnP+NeuS+BA           | 71.21       | -16.89        |
| Only_PnP              | 42.89       | -45.21        |

---

## 1. Ablation: no_rectified_by_hand

**Config:** `output[5_3_16_00][fb2b208][Ablation][no_rectified_by_hand]`
**Avg ADD AUC:** 88.10 → 82.79 (Δ = **-5.31**)

| Sequence | Baseline | Ablation | Δ ADD AUC |
|----------|----------|----------|-----------|
| BB13     | 85.19    | 49.41    | **-35.78** |
| GSF12    | 86.20    | 51.23    | **-34.97** |
| MC4      | 86.49    | 59.06    | **-27.43** |
| GPMF12   | 58.99    | 54.87    | -4.12     |
| BB12     | 70.91    | 66.80    | -4.11     |
| GPMF14   | 89.82    | 86.76    | -3.06     |
| MDF12    | 94.97    | 94.83    | -0.14     |
| SMu40    | 93.22    | 93.18    | -0.04     |
| ShSu10   | 92.14    | 92.12    | -0.02     |
| ABF12    | 92.54    | 92.55    | +0.01     |
| SMu1     | 96.23    | 96.30    | +0.07     |
| MC1      | 94.24    | 94.34    | +0.10     |
| SM4      | 94.79    | 95.02    | +0.23     |
| ABF14    | 88.26    | 89.12    | +0.86     |
| GSF13    | 90.76    | 91.65    | +0.89     |
| MDF14    | 94.76    | 96.36    | +1.60     |
| SM2      | 92.31    | 94.32    | +2.01     |
| ShSu12   | 84.02    | 92.21    | **+8.19** |
| **Avg**  | **88.10**| **82.79**| **-5.31** |

**Takeaway:** Hand rectification is critical for BB13, GSF12, and MC4 (−27 to −36 pts). ShSu12 is an outlier that benefits from removing it (+8.2 pts). 9/18 sequences regress.

---

## 2. Ablation: Only_PnP

**Config:** `output[4_30_21_36][f00feea][Ablation][Only_PnP]`
**Avg ADD AUC:** 88.10 → 42.89 (Δ = **-45.21**)

| Sequence | Baseline | Ablation | Δ ADD AUC |
|----------|----------|----------|-----------|
| BB13     | 85.19    | 9.34     | **-75.85** |
| GSF13    | 90.76    | 20.80    | **-69.96** |
| MDF12    | 94.97    | 27.94    | **-67.03** |
| MDF14    | 94.76    | 27.76    | **-67.00** |
| GSF12    | 86.20    | 20.03    | **-66.17** |
| BB12     | 70.91    | 9.32     | **-61.59** |
| ABF12    | 92.54    | 37.61    | -54.93    |
| SMu40    | 93.22    | 41.87    | -51.35    |
| SM4      | 94.79    | 49.03    | -45.76    |
| ShSu10   | 92.14    | 50.11    | -42.03    |
| ABF14    | 88.26    | 49.32    | -38.94    |
| SM2      | 92.31    | 54.63    | -37.68    |
| SMu1     | 96.23    | 63.37    | -32.86    |
| GPMF12   | 58.99    | 27.37    | -31.62    |
| GPMF14   | 89.82    | 59.00    | -30.82    |
| MC4      | 86.49    | 60.38    | -26.11    |
| ShSu12   | 84.02    | 75.83    | -8.19     |
| MC1      | 94.24    | 88.27    | -5.97     |
| **Avg**  | **88.10**| **42.89**| **-45.21** |

**Takeaway:** All 18 sequences regress without FoundationPose and NeuS. NeuS in particular provides large gains for long/complex sequences (MDF12/14, GSF13 drop >67 pts). The combination of FP + NeuS is essential for the pipeline.

---

## 3. Ablation: SAM3D+FP+BA (no NeuS)

**Config:** `output[5_1_8_39][3c57beb][Ablation][SAM3D+FP+BA]`
**Avg ADD AUC:** 88.10 → 85.18 (Δ = **-2.92**)

| Sequence | Baseline | Ablation | Δ ADD AUC |
|----------|----------|----------|-----------|
| GSF12    | 86.20    | 51.74    | **-34.46** |
| BB13     | 85.19    | 56.61    | **-28.58** |
| BB12     | 70.91    | 55.40    | -15.51    |
| MDF12    | 94.97    | 81.97    | -13.00    |
| ABF12    | 92.54    | 87.90    | -4.64     |
| MC4      | 86.49    | 85.18    | -1.31     |
| GSF13    | 90.76    | 89.53    | -1.23     |
| MDF14    | 94.76    | 94.33    | -0.43     |
| SM4      | 94.79    | 94.45    | -0.34     |
| ShSu10   | 92.14    | 91.86    | -0.28     |
| MC1      | 94.24    | 94.19    | -0.05     |
| SMu40    | 93.22    | 93.24    | +0.02     |
| SMu1     | 96.23    | 96.26    | +0.03     |
| ABF14    | 88.26    | 88.64    | +0.38     |
| SM2      | 92.31    | 94.72    | +2.41     |
| GPMF14   | 89.82    | 94.87    | +5.05     |
| ShSu12   | 84.02    | 91.57    | +7.55     |
| GPMF12   | 58.99    | 90.86    | **+31.87** |
| **Avg**  | **88.10**| **85.18**| **-2.92** |

**Takeaway:** Removing NeuS hurts GSF12 (−34) and BB13 (−29) most severely, but GPMF12 strongly benefits (+31.9) suggesting NeuS sometimes degrades results on sequences where the SAM3D+FP init is already strong. The average degradation is relatively mild (−2.92 pts).

---

## 4. Ablation: PnP+NeuS+BA (no FoundationPose)

**Config:** `output[5_1_22_17][c270f69][Ablation][PnP+Neus+BA]`
**Avg ADD AUC:** 88.10 → 71.21 (Δ = **-16.89**)

| Sequence | Baseline | Ablation | Δ ADD AUC |
|----------|----------|----------|-----------|
| BB13     | 85.19    | 16.57    | **-68.62** |
| SMu40    | 93.22    | 31.19    | **-62.03** |
| GPMF14   | 89.82    | 55.32    | **-34.50** |
| SMu1     | 96.23    | 61.86    | **-34.37** |
| SM4      | 94.79    | 60.45    | **-34.34** |
| GSF12    | 86.20    | 69.92    | -16.28    |
| GPMF12   | 58.99    | 46.87    | -12.12    |
| GSF13    | 90.76    | 78.93    | -11.83    |
| MC4      | 86.49    | 74.67    | -11.82    |
| BB12     | 70.91    | 60.61    | -10.30    |
| ABF12    | 92.54    | 85.13    | -7.41     |
| MC1      | 94.24    | 92.31    | -1.93     |
| ShSu10   | 92.14    | 91.22    | -0.92     |
| SM2      | 92.31    | 92.01    | -0.30     |
| MDF14    | 94.76    | 94.58    | -0.18     |
| MDF12    | 94.97    | 94.98    | +0.01     |
| ShSu12   | 84.02    | 85.04    | +1.02     |
| ABF14    | 88.26    | 90.15    | +1.89     |
| **Avg**  | **88.10**| **71.21**| **-16.89** |

**Takeaway:** FoundationPose is critical: removing it collapses BB13 (−69), SMu40 (−62), and causes severe drops on SMu1, SM4, GPMF14 (all ~−34). MDF12 is largely unaffected; ABF14 and ShSu12 slightly improve. 16/18 sequences regress.

---

## 5. Ablation: no_depth_filter

**Config:** `output[4_29_9_14][e781969][Ablation][no_depth_filter]`
**Avg ADD AUC:** 88.10 → 84.49 (Δ = **-3.61**)

| Sequence | Baseline | Ablation | Δ ADD AUC |
|----------|----------|----------|-----------|
| BB13     | 85.19    | 49.16    | **-36.03** |
| GSF12    | 86.20    | 50.78    | **-35.42** |
| GPMF14   | 89.82    | 74.76    | **-15.06** |
| MC1      | 94.24    | 93.79    | -0.45     |
| MDF12    | 94.97    | 94.60    | -0.37     |
| ABF12    | 92.54    | 92.38    | -0.16     |
| SM2      | 92.31    | 92.21    | -0.10     |
| SMu1     | 96.23    | 96.37    | +0.14     |
| SM4      | 94.79    | 94.95    | +0.16     |
| ShSu10   | 92.14    | 92.35    | +0.21     |
| SMu40    | 93.22    | 93.52    | +0.30     |
| ABF14    | 88.26    | 88.98    | +0.72     |
| GSF13    | 90.76    | 91.57    | +0.81     |
| MC4      | 86.49    | 87.71    | +1.22     |
| MDF14    | 94.76    | 96.27    | +1.51     |
| GPMF12   | 58.99    | 62.17    | +3.18     |
| ShSu12   | 84.02    | 91.13    | +7.11     |
| BB12     | 70.91    | 78.20    | **+7.29** |
| **Avg**  | **88.10**| **84.49**| **-3.61** |

**Takeaway:** Depth filtering is critical for BB13 (−36) and GSF12 (−35), likely due to noisy depth corrupting pose estimation on those sequences. GPMF14 also suffers (−15). Interestingly, BB12 and ShSu12 benefit notably (+7.3, +7.1) without depth filtering, suggesting the filter occasionally discards valid points. 10/18 sequences regress overall.
