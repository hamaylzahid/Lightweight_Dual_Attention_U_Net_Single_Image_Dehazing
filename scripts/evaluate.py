# ============================================================
# RESIDE-6K TEST EVALUATION
# EDGE-AWARE LIGHTWEIGHT DUAL ATTENTION U-NET
#
# METRICS:
#   PSNR
#   SSIM
#   MSE
#   MAE
#   Inference time
#   FPS
#
# EVALUATION:
#   1000 TEST paired images
#   256 x 256
#
# OUTPUTS:
#   CSV per-image metrics
#   JSON summary
#   Visual comparison samples
#   Best/worst images
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import os
import json
import time
import math
import csv
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from PIL import Image
import torchvision.transforms as T

from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from src.model import UNetDual
from src.dataset import create_test_pairs, create_test_loader
from src.metrics import ssim, calculate_psnr


# ============================================================
# 1. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

DATASET_DIR = Path(
    "/content/RESIDE-6K"
)

TEST_HAZY_DIR = (
    DATASET_DIR /
    "test" /
    "hazy"
)

TEST_GT_DIR = (
    DATASET_DIR /
    "test" /
    "GT"
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_DIR = Path(
    "/content/drive/MyDrive/dehazingdatasets (1)/RESIDE-6K_full_dual_model"
)

CHECKPOINT_DIR = (
    MODEL_DIR /
    "checkpoints"
)


# ------------------------------------------------------------
# USE BEST CHECKPOINT
# ------------------------------------------------------------

CHECKPOINT_PATH = (
    CHECKPOINT_DIR /
    "best_checkpoint.pth"
)


# ------------------------------------------------------------
# ALTERNATIVE:
# FINAL MODEL
# ------------------------------------------------------------

FINAL_MODEL_PATH = (
    CHECKPOINT_DIR /
    "final_model.pth"
)


# ------------------------------------------------------------
# EVALUATION OUTPUTS
# ------------------------------------------------------------

EVAL_DIR = (
    MODEL_DIR /
    "evaluation"
)

EVAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VIS_DIR = (
    EVAL_DIR /
    "visualizations"
)

VIS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CSV_PATH = (
    EVAL_DIR /
    "per_image_metrics.csv"
)

JSON_PATH = (
    EVAL_DIR /
    "evaluation_summary.json"
)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

IMAGE_SIZE = 256

BATCH_SIZE = 8

NUM_WORKERS = 2


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n" + "=" * 70)
print("EVALUATION DEVICE")
print("=" * 70)

print(
    "Device:",
    DEVICE
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# 3. VERIFY PATHS
# ============================================================

print("\n" + "=" * 70)
print("PATH VERIFICATION")
print("=" * 70)

assert DATASET_DIR.exists(), (
    f"Dataset not found:\n{DATASET_DIR}"
)

assert TEST_HAZY_DIR.exists(), (
    f"Test hazy directory not found:\n{TEST_HAZY_DIR}"
)

assert TEST_GT_DIR.exists(), (
    f"Test GT directory not found:\n{TEST_GT_DIR}"
)

assert CHECKPOINT_PATH.exists(), (
    f"Best checkpoint not found:\n{CHECKPOINT_PATH}"
)

print(
    "Dataset:",
    DATASET_DIR
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print(
    "Path verification: PASS"
)


# ============================================================
# 4. CREATE TEST PAIRS
# ============================================================

test_pairs = create_test_pairs(
    TEST_HAZY_DIR,
    TEST_GT_DIR
)


# ============================================================
# 5. TEST DATALOADER
# ============================================================

test_dataset, test_loader = create_test_loader(
    test_pairs,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS
)


print("\n" + "=" * 70)
print("TEST DATALOADER")
print("=" * 70)

print(
    "Test images:",
    len(test_dataset)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Batches:",
    len(test_loader)
)

print(
    "Resolution:",
    f"{IMAGE_SIZE}x{IMAGE_SIZE}"
)


# ============================================================
# 6. CREATE MODEL
# ============================================================

model = UNetDual().to(
    DEVICE
)


# ============================================================
# 7. PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

parameter_size_mb = (
    total_params *
    4 /
    (1024 ** 2)
)


print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(
    "Architecture:",
    "Edge-Aware Lightweight Dual Attention U-Net"
)

print(
    "Class:",
    "UNetDual"
)

print(
    f"Total parameters: {total_params:,}"
)

print(
    f"Trainable parameters: {trainable_params:,}"
)

print(
    f"Parameters: {total_params / 1e6:.6f} M"
)

print(
    f"FP32 parameter size: {parameter_size_mb:.2f} MB"
)


# ============================================================
# 8. LOAD BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("LOADING CHECKPOINT")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


assert (
    checkpoint.get("architecture")
    == "UNetDual"
), (
    "Checkpoint architecture mismatch."
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


checkpoint_epoch = checkpoint.get(
    "epoch",
    None
)

checkpoint_loss = checkpoint.get(
    "loss",
    None
)

checkpoint_best_loss = checkpoint.get(
    "best_loss",
    None
)


print(
    "Checkpoint epoch:",
    checkpoint_epoch
)

print(
    "Checkpoint loss:",
    checkpoint_loss
)

print(
    "Best training loss:",
    checkpoint_best_loss
)

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# 9. EVALUATION
# ============================================================

model.eval()


all_results = []


total_psnr = 0.0

total_ssim = 0.0

total_mse = 0.0

total_mae = 0.0

total_inference_time = 0.0

total_images = 0


print("\n" + "=" * 70)
print("STARTING TEST EVALUATION")
print("=" * 70)

print(
    "Test images:",
    len(test_dataset)
)

print(
    "Resolution:",
    f"{IMAGE_SIZE}x{IMAGE_SIZE}"
)

print(
    "Batch size:",
    BATCH_SIZE
)


with torch.inference_mode():

    for batch_idx, (
        hazy,
        gt,
        names
    ) in enumerate(
        tqdm(
            test_loader,
            desc="Evaluating",
            unit="batch"
        )
    ):

        hazy = hazy.to(
            DEVICE,
            non_blocking=True
        )

        gt = gt.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # GPU SYNCHRONIZATION
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        start_time = time.perf_counter()


        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        pred = model(
            hazy
        )


        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        inference_time = (
            time.perf_counter()
            -
            start_time
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        batch_psnr = calculate_psnr(
            pred,
            gt
        )

        batch_ssim = ssim(
            pred,
            gt
        )

        batch_mse = torch.mean(
            (pred - gt) ** 2,
            dim=(1, 2, 3)
        )

        batch_mae = torch.mean(
            torch.abs(
                pred - gt
            ),
            dim=(1, 2, 3)
        )


        batch_size_actual = (
            hazy.size(0)
        )


        inference_per_image = (
            inference_time /
            batch_size_actual
        )


        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        for i in range(
            batch_size_actual
        ):

            result = {

                "image":
                    names[i],

                "psnr_db":
                    float(
                        batch_psnr[i].item()
                    ),

                "ssim":
                    float(
                        batch_ssim[i].item()
                    ),

                "mse":
                    float(
                        batch_mse[i].item()
                    ),

                "mae":
                    float(
                        batch_mae[i].item()
                    ),

                "inference_time_ms":
                    float(
                        inference_per_image
                        *
                        1000
                    )

            }

            all_results.append(
                result
            )


        # ----------------------------------------------------
        # ACCUMULATE
        # ----------------------------------------------------

        total_psnr += (
            batch_psnr.sum().item()
        )

        total_ssim += (
            batch_ssim.sum().item()
        )

        total_mse += (
            batch_mse.sum().item()
        )

        total_mae += (
            batch_mae.sum().item()
        )

        total_inference_time += (
            inference_time
        )

        total_images += (
            batch_size_actual
        )


# ============================================================
# 10. FINAL METRICS
# ============================================================

average_psnr = (
    total_psnr /
    total_images
)

average_ssim = (
    total_ssim /
    total_images
)

average_mse = (
    total_mse /
    total_images
)

average_mae = (
    total_mae /
    total_images
)

average_inference_ms = (
    total_inference_time /
    total_images
    *
    1000
)

fps = (
    total_images /
    total_inference_time
)


# ============================================================
# 11. METRIC DISTRIBUTION
# ============================================================

psnr_values = np.array([
    x["psnr_db"]
    for x in all_results
])

ssim_values = np.array([
    x["ssim"]
    for x in all_results
])

mse_values = np.array([
    x["mse"]
    for x in all_results
])

mae_values = np.array([
    x["mae"]
    for x in all_results
])


summary = {

    "architecture":
        "Edge-Aware Lightweight Dual Attention U-Net",

    "model_class":
        "UNetDual",

    "checkpoint":
        str(CHECKPOINT_PATH),

    "checkpoint_epoch":
        checkpoint_epoch,

    "dataset":
        "RESIDE-6K",

    "test_images":
        total_images,

    "image_size":
        f"{IMAGE_SIZE}x{IMAGE_SIZE}",

    "parameters":
        int(total_params),

    "parameters_million":
        float(
            total_params / 1e6
        ),

    "parameter_size_mb_fp32":
        float(
            parameter_size_mb
        ),

    "PSNR_dB":
        float(
            average_psnr
        ),

    "SSIM":
        float(
            average_ssim
        ),

    "MSE":
        float(
            average_mse
        ),

    "MAE":
        float(
            average_mae
        ),

    "PSNR_std":
        float(
            np.std(
                psnr_values
            )
        ),

    "SSIM_std":
        float(
            np.std(
                ssim_values
            )
        ),

    "PSNR_min":
        float(
            np.min(
                psnr_values
            )
        ),

    "PSNR_max":
        float(
            np.max(
                psnr_values
            )
        ),

    "SSIM_min":
        float(
            np.min(
                ssim_values
            )
        ),

    "SSIM_max":
        float(
            np.max(
                ssim_values
            )
        ),

    "average_inference_ms":
        float(
            average_inference_ms
        ),

    "FPS":
        float(
            fps
        )

}


# ============================================================
# 12. SAVE CSV
# ============================================================

df = pd.DataFrame(
    all_results
)

df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# 13. SAVE JSON
# ============================================================

with open(
    JSON_PATH,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# 14. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("========== FINAL TEST RESULTS ==========")
print("=" * 70)

print(
    f"Test images:          {total_images:,}"
)

print(
    f"Resolution:           {IMAGE_SIZE}x{IMAGE_SIZE}"
)

print()

print(
    f"PSNR:                 {average_psnr:.4f} dB"
)

print(
    f"SSIM:                 {average_ssim:.6f}"
)

print(
    f"MSE:                  {average_mse:.8f}"
)

print(
    f"MAE:                  {average_mae:.8f}"
)

print()

print(
    f"PSNR std:             {np.std(psnr_values):.4f}"
)

print(
    f"SSIM std:             {np.std(ssim_values):.6f}"
)

print()

print(
    f"Minimum PSNR:         {np.min(psnr_values):.4f} dB"
)

print(
    f"Maximum PSNR:         {np.max(psnr_values):.4f} dB"
)

print()

print(
    f"Minimum SSIM:         {np.min(ssim_values):.6f}"
)

print(
    f"Maximum SSIM:         {np.max(ssim_values):.6f}"
)

print()

print(
    f"Avg inference:        {average_inference_ms:.3f} ms/image"
)

print(
    f"FPS:                  {fps:.2f}"
)

print()

print(
    f"Parameters:           {total_params:,}"
)

print(
    f"Parameters:           {total_params / 1e6:.6f} M"
)

print(
    f"FP32 parameter size:  {parameter_size_mb:.2f} MB"
)

print()

print(
    "Per-image CSV:"
)

print(
    CSV_PATH
)

print()

print(
    "Summary JSON:"
)

print(
    JSON_PATH
)


# ============================================================
# 15. BEST / WORST IMAGES
# ============================================================

best_psnr = df.nlargest(
    10,
    "psnr_db"
)

worst_psnr = df.nsmallest(
    10,
    "psnr_db"
)

best_ssim = df.nlargest(
    10,
    "ssim"
)

worst_ssim = df.nsmallest(
    10,
    "ssim"
)


print("\n" + "=" * 70)
print("TOP 10 PSNR")
print("=" * 70)

print(
    best_psnr[
        [
            "image",
            "psnr_db",
            "ssim",
            "mae"
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("BOTTOM 10 PSNR")
print("=" * 70)

print(
    worst_psnr[
        [
            "image",
            "psnr_db",
            "ssim",
            "mae"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 16. VISUALIZATION FUNCTION
# ============================================================

def load_image_tensor(path):

    image = Image.open(
        path
    ).convert("RGB")

    transform = T.Compose([

        T.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        T.ToTensor()

    ])

    return transform(
        image
    ).unsqueeze(0)


def generate_prediction(
    hazy_path
):

    x = load_image_tensor(
        hazy_path
    ).to(DEVICE)

    model.eval()

    with torch.inference_mode():

        pred = model(x)

    return pred[0].cpu()


def save_visualization(
    image_name,
    output_name
):

    hazy_path = (
        TEST_HAZY_DIR /
        image_name
    )

    gt_path = (
        TEST_GT_DIR /
        image_name
    )


    hazy = load_image_tensor(
        hazy_path
    )[0]

    gt = load_image_tensor(
        gt_path
    )[0]

    pred = generate_prediction(
        hazy_path
    )


    hazy_np = (
        hazy.permute(
            1,
            2,
            0
        ).numpy()
    )

    gt_np = (
        gt.permute(
            1,
            2,
            0
        ).numpy()
    )

    pred_np = (
        pred.permute(
            1,
            2,
            0
        ).numpy()
    )


    row = df[
        df["image"]
        ==
        image_name
    ].iloc[0]


    fig = plt.figure(
        figsize=(15, 5)
    )


    ax1 = plt.subplot(
        1,
        3,
        1
    )

    ax1.imshow(
        hazy_np
    )

    ax1.set_title(
        "Hazy Input"
    )

    ax1.axis(
        "off"
    )


    ax2 = plt.subplot(
        1,
        3,
        2
    )

    ax2.imshow(
        pred_np
    )

    ax2.set_title(
        f"Prediction\n"
        f"PSNR={row['psnr_db']:.2f} dB | "
        f"SSIM={row['ssim']:.4f}"
    )

    ax2.axis(
        "off"
    )


    ax3 = plt.subplot(
        1,
        3,
        3
    )

    ax3.imshow(
        gt_np
    )

    ax3.set_title(
        "Ground Truth"
    )

    ax3.axis(
        "off"
    )


    plt.tight_layout()

    save_path = (
        VIS_DIR /
        output_name
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(
        fig
    )


# ============================================================
# 17. VISUALIZE BEST PSNR
# ============================================================

print("\n" + "=" * 70)
print("BEST PSNR EXAMPLES")
print("=" * 70)

for i, image_name in enumerate(
    best_psnr["image"].head(5),
    start=1
):

    save_visualization(
        image_name,
        f"best_psnr_{i:02d}.png"
    )


# ============================================================
# 18. VISUALIZE WORST PSNR
# ============================================================

print("\n" + "=" * 70)
print("WORST PSNR EXAMPLES")
print("=" * 70)

for i, image_name in enumerate(
    worst_psnr["image"].head(5),
    start=1
):

    save_visualization(
        image_name,
        f"worst_psnr_{i:02d}.png"
    )


# ============================================================
# 19. FINAL CHECKS
# ============================================================

assert len(df) == 1000

assert np.isfinite(
    average_psnr
)

assert np.isfinite(
    average_ssim
)

assert np.isfinite(
    average_mse
)

assert np.isfinite(
    average_mae
)

assert 0 <= average_ssim <= 1

assert average_mse >= 0

assert average_mae >= 0

assert CSV_PATH.exists()

assert JSON_PATH.exists()


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(
    "Evaluation verification: PASS"
)

print()

print(
    "CSV:",
    CSV_PATH
)

print(
    "JSON:",
    JSON_PATH
)

print(
    "Visualizations:",
    VIS_DIR
)