<h1 align="center">Edge-Aware Lightweight Dual Attention U-Net for Image Dehazing<br></h1>

<p align="center">
A lightweight U-Net for single-image dehazing, combining depthwise-separable convolutional blocks with bottleneck channel-spatial dual attention, achieving 25.23 dB PSNR and 0.9056 SSIM with 1.08M parameters.
</p>

<p align="center">
<em>Research Project | Computer Vision | Image Restoration</em>
</p>

<p align="center">
<img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
<img src="https://img.shields.io/badge/torch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/torchvision-EE4C2C?logo=pytorch&logoColor=white" alt="torchvision">
<img src="https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Pillow-yellow" alt="Pillow">
<img src="https://img.shields.io/badge/tqdm-yellow" alt="tqdm">
<img src="https://img.shields.io/badge/matplotlib-11557C" alt="matplotlib">
<img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas">
</p>

---

<h2 align="center">Abstract<br></h2>

Single-image dehazing addresses the loss of visibility, contrast, and detail caused by atmospheric scattering, relevant to autonomous perception, aerial imaging, surveillance, and outdoor visual analysis. This project develops a lightweight encoder-decoder restoration network with dual attention (channel attention followed by spatial attention) applied at the 512-channel bottleneck, trained on RESIDE-6K with an L1 reconstruction objective.

Evaluated on 1,000 held-out test pairs at 256x256 resolution, the model reaches 25.23 dB PSNR and 0.9056 SSIM at 1.08M parameters, with inference at 158 FPS on a single Tesla T4.

| Metric | Result |
|---|---:|
| PSNR | 25.2250 dB |
| SSIM | 0.905627 |
| MSE | 0.00461821 |
| MAE | 0.04537791 |
| Inference time | 6.312 ms/image |
| Throughput | 158.42 FPS |
| Parameters | 1,082,629 (1.08M) |
| FP32 size | 4.13 MB |

---

<h2 align="center">Motivation<br></h2>

A dehazing model that recovers scene appearance well but is too heavy or slow to deploy has limited practical use. This project asks a narrower question: how much restoration quality can a sub-1.1M-parameter model recover when attention is concentrated at the bottleneck rather than distributed throughout the network? Concentrating attention at the bottleneck limits the additional attention modules to a single high-level feature representation while allowing the network to recalibrate informative channels and spatial regions, keeping the overall parameter and compute budget low.

---

<h2 align="center">Architecture<br></h2>

```
Input (3x256x256)
   |
Encoder 1 (3->64) -> pool
   |
Encoder 2 (64->128) -> pool
   |
Encoder 3 (128->256) -> pool
   |
Encoder 4 (256->512) -> pool
   |
Dual Attention (Channel -> Spatial), 512 channels
   |
Decoder 1 (512->256) + skip from Encoder 3
   |
Decoder 2 (256->128) + skip from Encoder 2
   |
Decoder 3 (128->64) + skip from Encoder 1
   |
1x1 Conv -> Sigmoid -> Output (3x256x256)
```

**Convolutional block:** 1x1 conv -> BatchNorm -> ReLU -> 3x3 depthwise conv -> BatchNorm -> 1x1 conv -> BatchNorm -> ReLU. The depthwise step reduces the cost of spatial feature extraction while keeping channel-wise processing intact.

**Channel attention:** global average pooling and global max pooling, passed through a shared MLP, combined and passed through sigmoid to produce a per-channel recalibration mask.

**Spatial attention:** channel-wise mean and max are concatenated and passed through a 7x7 convolution and sigmoid to produce a spatial recalibration mask.

| Component | Configuration |
|---|---|
| Model class | `UNetDual` |
| Encoder channels | 64 -> 128 -> 256 -> 512 |
| Bottleneck | 512, channel + spatial attention |
| Decoder channels | 256 -> 128 -> 64 |
| Output activation | Sigmoid |
| Total parameters | 1,082,629 |
| FP32 size | 4.13 MB |

---

<h2 align="center">Dataset<br></h2>

RESIDE-6K, paired hazy/ground-truth images, explicitly verified by filename before use.

| Split | Hazy | GT | Matched pairs | Unmatched |
|---|---:|---:|---:|---:|
| Train | 6,000 | 6,000 | 6,000 | 0 |
| Test | 1,000 | 1,000 | 1,000 | 0 |

<h2 align="center">Training<br></h2>

| Setting | Value |
|---|---|
| Image size | 256x256 |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Loss | L1 (mean absolute reconstruction error) |
| Checkpointing | latest + best, auto-resume from last completed epoch |
| Final training loss | 0.03793 (epoch 50) |

Checkpoints store model state, optimizer state, epoch, current/best loss, parameter counts, and full training history, so an interrupted run resumes without restarting.

<h2 align="center">Evaluation Protocol<br></h2>

1,000 test pairs, 256x256, batch size 8, evaluated on an NVIDIA Tesla T4 (CUDA 12.8, 14.56 GB VRAM) with synchronized GPU timing. Per-image PSNR, SSIM, MSE, MAE, and inference time are logged individually, then aggregated into mean, standard deviation, and min/max, alongside FPS and parameter count. Checkpoint architecture is verified (`UNetDual`) before evaluation, and result validity is checked (finite metrics, SSIM in [0,1], MSE/MAE >= 0) before results are written.

<h2 align="center">Results<br></h2>

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| PSNR (dB) | 25.2250 | 4.0837 | 14.6240 | 36.5805 |
| SSIM | 0.9056 | 0.0498 | 0.7346 | 0.9789 |

**Highest-PSNR case:** 36.58 dB PSNR / 0.975 SSIM
**Lowest-PSNR case:** 14.62 dB PSNR / 0.799 SSIM

The lowest-scoring images correspond to challenging heavy-haze cases in the evaluated test set. Per-image results (CSV) and aggregate results (JSON) are both retained, along with saved visualizations of the five best and five worst PSNR cases, so the model isn't evaluated only on favorable examples.

<h2 align="center">Efficiency<br></h2>

| Metric | Result |
|---|---:|
| Inference time | 6.312 ms/image |
| Throughput | 158.42 FPS |
| Parameters | 1.08M |
| FP32 size | 4.13 MB |

Measured on a Tesla T4 at 256x256; not a hardware-independent figure.

---

<h2 align="center">Repository Structure<br></h2>

Tracked source files:

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── model.py         # conv_block, ChannelAttention, SpatialAttention, DualAttention, UNetDual
│   ├── dataset.py        # pairing, verification, dataset classes
│   └── metrics.py        # PSNR, SSIM
└── scripts/
    ├── train.py           # training loop, checkpoint auto-resume
    └── evaluate.py        # test-set evaluation, per-image metrics, visualizations
```

Generated artifacts (produced by running `train.py` / `evaluate.py`, not committed to the repo):

```
checkpoints/
├── best_checkpoint.pth
└── latest_checkpoint.pth

evaluation/
├── per_image_metrics.csv
├── evaluation_summary.json
└── visualizations/
```

---

<h2 align="center">Reproducibility<br></h2>

The training and evaluation pipelines are implemented as standalone Python scripts. Training automatically resumes from the latest checkpoint when available, while evaluation verifies checkpoint architecture and dataset pairing before computing metrics.

The reported experiment uses:

- 6,000 paired training images
- 1,000 paired test images
- 256x256 input resolution
- batch size 8
- 50 training epochs
- Adam optimizer
- learning rate 1e-4
- L1 reconstruction loss
- NVIDIA Tesla T4 GPU

See `requirements.txt` for the Python dependencies.

---

<h2 align="center">Limitations<br></h2>

**Checkpoint selection.** The best checkpoint is selected by lowest training L1 loss, not validation PSNR/SSIM. Validation-driven selection is a natural next step.

**Single-dataset evaluation.** Primary quantitative results are on the RESIDE-6K test split. Cross-dataset evaluation (e.g. NH-HAZE) would give a stronger measure of generalization to real-world haze.

**No ablation yet.** The contribution of channel attention and spatial attention individually has not been isolated. A baseline-vs-channel-only-vs-full-dual-attention comparison, under the same training protocol, is the most direct way to attribute the performance gain to the attention mechanism rather than to the architecture as a whole.

<h2 align="center">Future Directions<br></h2>

- Ablation isolating channel attention and spatial attention contributions
- Cross-dataset generalization testing (NH-HAZE, real-world haze)
- Validation-based checkpoint selection
- Evaluation at higher input resolutions
- Failure-case analysis on the densest-haze samples

---

<h2 align="center">License<br></h2>

<p align="center">
<img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT">
</p>

This project is released under the MIT License.

<h2 align="center">Dependencies<br></h2>

<p align="center">
<img src="https://img.shields.io/badge/torch-EE4C2C?logo=pytorch&logoColor=white" alt="torch">
<img src="https://img.shields.io/badge/torchvision-EE4C2C?logo=pytorch&logoColor=white" alt="torchvision">
<img src="https://img.shields.io/badge/numpy-013243?logo=numpy&logoColor=white" alt="numpy">
<img src="https://img.shields.io/badge/Pillow-yellow" alt="Pillow">
<img src="https://img.shields.io/badge/tqdm-yellow" alt="tqdm">
<img src="https://img.shields.io/badge/matplotlib-11557C" alt="matplotlib">
<img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" alt="pandas">
</p>

See `requirements.txt` for pinned versions.

<h2 align="center">Author<br></h2>

<p align="center">
  <strong>Hamayl Zahid</strong>
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/hamaylzahid/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://hamaylzahid.github.io/hamaylzahid/">
    <img src="https://img.shields.io/badge/Portfolio-000000?logo=googlechrome&logoColor=white" alt="Portfolio">
  </a>
</p>
