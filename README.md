# TugasSHP

### Output

PS C:\Kampus\SHP kecerdasan\topik9-10> python -u "c:\Kampus\SHP kecerdasan\topik9-10\tempCodeRunnerFile.py"
DATASET XOR:
Input (X1, X2) -> Output (Y)
(0, 0) -> 0
(0, 1) -> 1
(1, 0) -> 1
(1, 1) -> 0
=== TRAINING NEURAL NETWORK XOR - ABSEN 1 ===
Arsitektur: 2 Input -> 3 Hidden -> 1 Output
Learning Rate: 0.1

Bobot Awal:
W1 (Input ke Hidden):
[[0.2 0.4 0.1]
 [0.3 0.1 0.5]]
W2 (Hidden ke Output):
[[0.6]
 [0.3]
 [0.4]]
Bias Hidden: [[0.1 0.2 0.1]]
Bias Output: [[0.2]]

==================================================

=== EPOCH 1 - DETAIL ===

Pattern 1: (0, 0) -> 0
Hidden nets: [0.100, 0.200, 0.100]
Hidden outputs: [0.525, 0.550, 0.525]
Output: 0.709
Error: -0.709
MSE: 0.251

Pattern 2: (0, 1) -> 1
Hidden nets: [0.389, 0.295, 0.593]
Hidden outputs: [0.596, 0.573, 0.644]
Output: 0.700
Error: 0.300
MSE: 0.045

Pattern 3: (1, 0) -> 1
Hidden nets: [0.293, 0.597, 0.195]
Hidden outputs: [0.573, 0.645, 0.549]
Output: 0.707
Error: 0.293
MSE: 0.043

Pattern 4: (1, 1) -> 0
Hidden nets: [0.606, 0.702, 0.703]
Hidden outputs: [0.647, 0.669, 0.669]
Output: 0.739
Error: -0.739
MSE: 0.273

Total MSE Epoch 1: 0.612092
Bobot setelah update:
W1: [[0.19408147 0.39700364 0.09625744]
 [0.29398146 0.09705181 0.49597911]]
W2: [0.54964461 0.24770097 0.34875101]

=== EPOCH 2 - DETAIL ===

Pattern 1: (0, 0) -> 0
Hidden nets: [0.088, 0.194, 0.092]
Hidden outputs: [0.522, 0.548, 0.523]
Output: 0.673
Error: -0.673
MSE: 0.226

Pattern 2: (0, 1) -> 1
Hidden nets: [0.372, 0.287, 0.582]
Hidden outputs: [0.592, 0.571, 0.641]
Output: 0.662
Error: 0.338
MSE: 0.057

Pattern 3: (1, 0) -> 1
Hidden nets: [0.277, 0.588, 0.185]
Hidden outputs: [0.569, 0.643, 0.546]
Output: 0.672
Error: 0.328
MSE: 0.054

Pattern 4: (1, 1) -> 0
Hidden nets: [0.583, 0.691, 0.688]
Hidden outputs: [0.642, 0.666, 0.666]
Output: 0.705
Error: -0.705
MSE: 0.249

Total MSE Epoch 2: 0.585569
Bobot setelah update:
W1: [[0.18941942 0.3947983  0.09344341]
 [0.2892127  0.09485636 0.49287272]]
W2: [0.50795917 0.20426481 0.30624578]

=== EPOCH 3 - DETAIL ===

Pattern 1: (0, 0) -> 0
Hidden nets: [0.078, 0.189, 0.085]
Hidden outputs: [0.519, 0.547, 0.521]
Output: 0.641
Error: -0.641
MSE: 0.205

Pattern 2: (0, 1) -> 1
Hidden nets: [0.359, 0.281, 0.573]
Hidden outputs: [0.589, 0.570, 0.640]
Output: 0.629
Error: 0.371
MSE: 0.069

Pattern 3: (1, 0) -> 1
Hidden nets: [0.263, 0.582, 0.176]
Hidden outputs: [0.565, 0.642, 0.544]
Output: 0.641
Error: 0.359
MSE: 0.064

Pattern 4: (1, 1) -> 0
Hidden nets: [0.566, 0.682, 0.677]
Hidden outputs: [0.638, 0.664, 0.663]
Output: 0.675
Error: -0.675
MSE: 0.228

Total MSE Epoch 3: 0.566461
Bobot setelah update:
W1: [[0.18573233 0.39315417 0.09131307]
 [0.28540739 0.09317906 0.49043379]]
W2: [0.47370634 0.16848159 0.27127423]
Epoch 4: Total MSE = 0.553131
Epoch 5: Total MSE = 0.544039
Epoch 6: Total MSE = 0.537931
Epoch 7: Total MSE = 0.533870
Epoch 8: Total MSE = 0.531189
Epoch 9: Total MSE = 0.529426
Epoch 10: Total MSE = 0.528271
Epoch 11: Total MSE = 0.527516
Epoch 12: Total MSE = 0.527023
Epoch 13: Total MSE = 0.526700
Epoch 14: Total MSE = 0.526489
Epoch 15: Total MSE = 0.526350

==================================================
HASIL AKHIR:

Testing Network:
Input -> Target | Predicted | Error

---

(0, 0) -> 0 | 0.5006 | 0.5006
(0, 1) -> 1 | 0.5101 | 0.4899
(1, 0) -> 1 | 0.5054 | 0.4946
(1, 1) -> 0 | 0.5147 | 0.5147

Akurasi rata-rata: 50.00%

Bobot Akhir:
W1 (Input ke Hidden):
[[0.16982749 0.38566615 0.0827399 ]
 [0.26713039 0.08287855 0.47719949]]
W2 (Hidden ke Output):
[0.33580617 0.02424671 0.13114008]
