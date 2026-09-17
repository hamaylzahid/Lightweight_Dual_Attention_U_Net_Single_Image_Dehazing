# ============================================================
# RESIDE-6K FULL TRAINING
# EDGE-AWARE LIGHTWEIGHT DUAL ATTENTION U-NET
#
# DATASET:
#   Source  -> Google Drive
#   Training -> /content/RESIDE-6K (FAST LOCAL STORAGE)
#
# MODEL:
#   Checkpoints/results/logs -> Google Drive
#
# TRAINING:
#   6000 TRAIN paired images
#   1000 TEST paired images
#   50 epochs
#   Batch size = 8
#   LR = 1e-4
#   Image size = 256
#   SAVE_PERIOD = -1
#   AUTO-RESUME
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

import json
import time
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from src.model import UNetDual
from src.dataset import (
    get_image_files,
    audit_split,
    PairedDataset,
)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# ORIGINAL DATASET ON GOOGLE DRIVE
# ------------------------------------------------------------

SOURCE_DATASET_DIR = Path(
    "/content/drive/MyDrive/dehazingdatasets (1)/RESIDE-6K"
)


# ------------------------------------------------------------
# LOCAL DATASET FOR FAST TRAINING
# ------------------------------------------------------------

LOCAL_DATASET_DIR = Path(
    "/content/RESIDE-6K"
)


# ------------------------------------------------------------
# MODEL OUTPUTS ON GOOGLE DRIVE
# ------------------------------------------------------------

MODEL_DIR = Path(
    "/content/drive/MyDrive/dehazingdatasets (1)/RESIDE-6K_full_dual_model"
)

CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

RESULTS_DIR = MODEL_DIR / "results"

LOGS_DIR = MODEL_DIR / "logs"


CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

EPOCHS = 50

IMAGE_SIZE = 256

SAVE_PERIOD = -1

NUM_WORKERS = 2


# ============================================================
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("\n" + "=" * 70)
print("DEVICE")
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
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

else:

    print(
        "WARNING: CUDA is not available."
    )


# ============================================================
# 5. VERIFY SOURCE DATASET
# ============================================================

print("\n" + "=" * 70)
print("SOURCE DATASET VERIFICATION")
print("=" * 70)

print(
    "Source:",
    SOURCE_DATASET_DIR
)

assert SOURCE_DATASET_DIR.exists(), (
    f"Source dataset not found:\n{SOURCE_DATASET_DIR}"
)

assert (
    SOURCE_DATASET_DIR / "train"
).exists(), (
    "Source TRAIN directory not found."
)

assert (
    SOURCE_DATASET_DIR / "test"
).exists(), (
    "Source TEST directory not found."
)

assert (
    SOURCE_DATASET_DIR / "train" / "hazy"
).exists(), (
    "Source TRAIN/hazy directory not found."
)

assert (
    SOURCE_DATASET_DIR / "train" / "GT"
).exists(), (
    "Source TRAIN/GT directory not found."
)

assert (
    SOURCE_DATASET_DIR / "test" / "hazy"
).exists(), (
    "Source TEST/hazy directory not found."
)

assert (
    SOURCE_DATASET_DIR / "test" / "GT"
).exists(), (
    "Source TEST/GT directory not found."
)

print(
    "Source dataset directories: PASS"
)


# ============================================================
# 6. SOURCE DATASET AUDIT
# ============================================================

(
    source_train_hazy,
    source_train_gt,
    source_train_matching,
    source_train_hazy_only,
    source_train_gt_only
) = audit_split(
    SOURCE_DATASET_DIR,
    "train"
)


(
    source_test_hazy,
    source_test_gt,
    source_test_matching,
    source_test_hazy_only,
    source_test_gt_only
) = audit_split(
    SOURCE_DATASET_DIR,
    "test"
)


# ============================================================
# 7. STRICT SOURCE DATASET ASSERTIONS
# ============================================================

assert len(source_train_hazy) == 6000, (
    f"Expected 6000 TRAIN hazy images, "
    f"found {len(source_train_hazy)}"
)

assert len(source_train_gt) == 6000, (
    f"Expected 6000 TRAIN GT images, "
    f"found {len(source_train_gt)}"
)

assert len(source_train_matching) == 6000, (
    f"Expected 6000 TRAIN matching pairs, "
    f"found {len(source_train_matching)}"
)

assert len(source_train_hazy_only) == 0, (
    "TRAIN contains hazy-only files."
)

assert len(source_train_gt_only) == 0, (
    "TRAIN contains GT-only files."
)


assert len(source_test_hazy) == 1000, (
    f"Expected 1000 TEST hazy images, "
    f"found {len(source_test_hazy)}"
)

assert len(source_test_gt) == 1000, (
    f"Expected 1000 TEST GT images, "
    f"found {len(source_test_gt)}"
)

assert len(source_test_matching) == 1000, (
    f"Expected 1000 TEST matching pairs, "
    f"found {len(source_test_matching)}"
)

assert len(source_test_hazy_only) == 0, (
    "TEST contains hazy-only files."
)

assert len(source_test_gt_only) == 0, (
    "TEST contains GT-only files."
)


print("\n" + "=" * 70)
print("SOURCE DATASET VERIFICATION")
print("=" * 70)

print("========== TRAIN ==========")
print("Hazy:       6000")
print("GT:         6000")
print("Matching:   6000")
print("Hazy only:     0")
print("GT only:       0")

print()

print("========== TEST ==========")
print("Hazy:       1000")
print("GT:         1000")
print("Matching:   1000")
print("Hazy only:     0")
print("GT only:       0")

print(
    "\nSource dataset verification: PASS"
)


# ============================================================
# 8. COPY DATASET TO /content
# ============================================================

print("\n" + "=" * 70)
print("LOCAL DATASET COPY")
print("=" * 70)

print(
    "Google Drive source:",
    SOURCE_DATASET_DIR
)

print(
    "Local training copy:",
    LOCAL_DATASET_DIR
)


local_copy_complete = (
    (LOCAL_DATASET_DIR / "train" / "hazy").exists()
    and
    (LOCAL_DATASET_DIR / "train" / "GT").exists()
    and
    (LOCAL_DATASET_DIR / "test" / "hazy").exists()
    and
    (LOCAL_DATASET_DIR / "test" / "GT").exists()
)


if local_copy_complete:

    print(
        "\nLocal dataset already exists."
    )

    print(
        "Skipping copy."
    )

else:

    if LOCAL_DATASET_DIR.exists():

        print(
            "\nIncomplete local dataset found."
        )

        print(
            "Removing incomplete copy..."
        )

        shutil.rmtree(
            LOCAL_DATASET_DIR
        )

    print(
        "\nCopying RESIDE-6K from Google Drive"
    )

    print(
        "to local Colab storage..."
    )

    copy_start = time.time()

    shutil.copytree(
        SOURCE_DATASET_DIR,
        LOCAL_DATASET_DIR
    )

    copy_time = (
        time.time() -
        copy_start
    )

    print(
        "\nDataset copy completed."
    )

    print(
        "Copy time:",
        round(
            copy_time / 60,
            2
        ),
        "minutes"
    )


# ============================================================
# 9. ACTIVE DATASET
# ============================================================

DATASET_DIR = LOCAL_DATASET_DIR


print("\n" + "=" * 70)
print("ACTIVE TRAINING DATASET")
print("=" * 70)

print(
    "Training will READ from:",
    DATASET_DIR
)

print(
    "This means image loading uses /content,",
    "not Google Drive."
)


# ============================================================
# 10. VERIFY LOCAL COPY
# ============================================================

assert DATASET_DIR.exists()

assert (
    DATASET_DIR / "train" / "hazy"
).exists()

assert (
    DATASET_DIR / "train" / "GT"
).exists()

assert (
    DATASET_DIR / "test" / "hazy"
).exists()

assert (
    DATASET_DIR / "test" / "GT"
).exists()


# ============================================================
# 11. LOCAL DATASET AUDIT
# ============================================================

(
    train_hazy_files,
    train_gt_files,
    train_matching,
    train_hazy_only,
    train_gt_only
) = audit_split(
    DATASET_DIR,
    "train"
)


(
    test_hazy_files,
    test_gt_files,
    test_matching,
    test_hazy_only,
    test_gt_only
) = audit_split(
    DATASET_DIR,
    "test"
)


# ============================================================
# 12. FINAL LOCAL ASSERTIONS
# ============================================================

assert len(train_hazy_files) == 6000
assert len(train_gt_files) == 6000
assert len(train_matching) == 6000
assert len(train_hazy_only) == 0
assert len(train_gt_only) == 0

assert len(test_hazy_files) == 1000
assert len(test_gt_files) == 1000
assert len(test_matching) == 1000
assert len(test_hazy_only) == 0
assert len(test_gt_only) == 0


print("\n" + "=" * 70)
print("LOCAL DATASET VERIFICATION: PASS")
print("=" * 70)


# ============================================================
# 13. BUILD PAIRED DATASET
# ============================================================

train_pairs = [

    (
        train_hazy_files[name],
        train_gt_files[name]
    )

    for name in sorted(
        train_matching
    )

]


test_pairs = [

    (
        test_hazy_files[name],
        test_gt_files[name]
    )

    for name in sorted(
        test_matching
    )

]


assert len(train_pairs) == 6000

assert len(test_pairs) == 1000


# ============================================================
# 14. CREATE TRAIN DATASET
# ============================================================

train_dataset = PairedDataset(
    train_pairs,
    IMAGE_SIZE
)


# ============================================================
# 15. CREATE TRAIN DATALOADER
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available(),

    drop_last=False
)


print("\n" + "=" * 70)
print("DATALOADER")
print("=" * 70)

print(
    "Dataset location:",
    DATASET_DIR
)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Batches per epoch:",
    len(train_loader)
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Total training steps:",
    len(train_loader) * EPOCHS
)


# ============================================================
# 16. CREATE FINAL TRAINING MODEL
# ============================================================

model = UNetDual().to(
    DEVICE
)


# ============================================================
# 17. PARAMETER VERIFICATION
# ============================================================

actual_total_params = sum(
    p.numel()
    for p in model.parameters()
)


actual_trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


non_trainable_params = (
    actual_total_params -
    actual_trainable_params
)


parameter_size_mb = (
    actual_total_params *
    4 /
    (1024 ** 2)
)


print("\n" + "=" * 70)
print("MODEL PARAMETER VERIFICATION")
print("=" * 70)

print(
    "Architecture:",
    "UNetDual"
)

print(
    "Attention:",
    "ChannelAttention + SpatialAttention"
)

print(
    "Attention location:",
    "512-channel bottleneck"
)

print(
    f"Total parameters:   {actual_total_params:,}"
)

print(
    f"Trainable:          {actual_trainable_params:,}"
)

print(
    f"Non-trainable:      {non_trainable_params:,}"
)

print(
    f"Parameters (M):     "
    f"{actual_total_params / 1e6:.6f}"
)

print(
    f"Parameter size:     "
    f"{parameter_size_mb:.2f} MB"
)


assert actual_total_params > 0

assert (
    actual_trainable_params ==
    actual_total_params
)


print(
    "\nParameter verification: PASS"
)


# ============================================================
# 18. LOSS + OPTIMIZER
# ============================================================

criterion = nn.L1Loss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 19. CHECKPOINT PATHS
# ============================================================

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR /
    "latest_checkpoint.pth"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR /
    "best_checkpoint.pth"
)

FINAL_MODEL = (
    CHECKPOINT_DIR /
    "final_model.pth"
)

HISTORY_FILE = (
    LOGS_DIR /
    "training_history.json"
)


# ============================================================
# 20. AUTO-RESUME
# ============================================================

start_epoch = 1

best_loss = float(
    "inf"
)

history = []


if LATEST_CHECKPOINT.exists():

    print("\n" + "=" * 70)
    print("CHECKPOINT FOUND")
    print("=" * 70)

    print(
        "Loading:",
        LATEST_CHECKPOINT
    )

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location=DEVICE
    )


    assert (
        checkpoint.get(
            "architecture"
        ) == "UNetDual"
    ), (
        "Checkpoint architecture does not match UNetDual."
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )


    last_epoch = checkpoint[
        "epoch"
    ]


    best_loss = checkpoint.get(
        "best_loss",
        checkpoint.get(
            "loss",
            float("inf")
        )
    )


    history = checkpoint.get(
        "history",
        []
    )


    start_epoch = (
        last_epoch + 1
    )


    print(
        "Previous epoch:",
        last_epoch
    )

    print(
        "Resuming from epoch:",
        start_epoch
    )

    print(
        f"Best loss: {best_loss:.6f}"
    )

else:

    print("\n" + "=" * 70)
    print("NO CHECKPOINT FOUND")
    print("=" * 70)

    print(
        "Starting from epoch 1."
    )


# ============================================================
# 21. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("========== TRAIN ==========")
print("=" * 70)

print("Hazy:       6000")
print("GT:         6000")
print("Matching:   6000")
print("Hazy only:  0")
print("GT only:    0")


print("\n" + "=" * 70)
print("========== TEST ==========")
print("=" * 70)

print("Hazy:       1000")
print("GT:         1000")
print("Matching:   1000")
print("Hazy only:  0")
print("GT only:    0")


print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(
    "Architecture:",
    "Edge-Aware Lightweight Dual Attention U-Net"
)

print(
    "Actual model class:",
    "UNetDual"
)

print(
    "Total parameters:",
    f"{actual_total_params:,}"
)

print(
    "Parameters (M):",
    f"{actual_total_params / 1e6:.6f}"
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Image size:",
    f"{IMAGE_SIZE}x{IMAGE_SIZE}"
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Save period:",
    SAVE_PERIOD
)

print(
    "Training data:",
    DATASET_DIR
)

print(
    "Checkpoint directory:",
    CHECKPOINT_DIR
)


# ============================================================
# 22. TRAINING
# ============================================================

if start_epoch > EPOCHS:

    print("\n" + "=" * 70)
    print("TRAINING ALREADY COMPLETED")
    print("=" * 70)

    print(
        "Last completed epoch:",
        start_epoch - 1
    )

    print(
        "Current EPOCHS:",
        EPOCHS
    )

else:

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    training_start = time.time()


    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):

        epoch_start = time.time()

        model.train()

        running_loss = 0.0


        progress_bar = tqdm(

            train_loader,

            total=len(train_loader),

            desc=(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS}"
            ),

            unit="batch",

            leave=True
        )


        # ====================================================
        # BATCH LOOP
        # ====================================================

        for batch_idx, (
            hazy,
            gt,
            _
        ) in enumerate(
            progress_bar,
            start=1
        ):

            hazy = hazy.to(
                DEVICE,
                non_blocking=True
            )

            gt = gt.to(
                DEVICE,
                non_blocking=True
            )


            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            pred = model(
                hazy
            )


            # ------------------------------------------------
            # LOSS
            # ------------------------------------------------

            loss = criterion(
                pred,
                gt
            )


            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()


            # ------------------------------------------------
            # RUNNING LOSS
            # ------------------------------------------------

            running_loss += (
                loss.item()
            )

            current_avg_loss = (
                running_loss /
                batch_idx
            )


            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            progress_bar.set_postfix({

                "loss":
                    f"{loss.item():.6f}",

                "avg":
                    f"{current_avg_loss:.6f}"

            })


        # ====================================================
        # EPOCH SUMMARY
        # ====================================================

        epoch_loss = (
            running_loss /
            len(train_loader)
        )

        epoch_time = (
            time.time() -
            epoch_start
        )


        history.append({

            "epoch":
                epoch,

            "loss":
                epoch_loss,

            "time_seconds":
                epoch_time

        })


        print(
            f"\nEpoch {epoch:03d}/{EPOCHS} | "
            f"Loss: {epoch_loss:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )


        # ====================================================
        # BEST MODEL
        # ====================================================

        is_best = (
            epoch_loss <
            best_loss
        )


        if is_best:

            best_loss = epoch_loss

            print(
                f"  New best model. "
                f"Best loss: {best_loss:.6f}"
            )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        checkpoint = {

            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "loss":
                epoch_loss,

            "best_loss":
                best_loss,

            "parameters":
                actual_total_params,

            "trainable_parameters":
                actual_trainable_params,

            "architecture":
                "UNetDual",

            "architecture_name":
                "Edge-Aware Lightweight Dual Attention U-Net",

            "attention":
                "ChannelAttention + SpatialAttention",

            "attention_location":
                "512-channel bottleneck",

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "image_size":
                IMAGE_SIZE,

            "epochs":
                EPOCHS,

            "save_period":
                SAVE_PERIOD,

            "dataset":
                "RESIDE-6K",

            "train_pairs":
                len(train_pairs),

            "test_pairs":
                len(test_pairs),

            "history":
                history

        }


        # ====================================================
        # SAVE LATEST
        # ====================================================

        torch.save(
            checkpoint,
            LATEST_CHECKPOINT
        )


        # ====================================================
        # SAVE BEST
        # ====================================================

        if is_best:

            torch.save(
                checkpoint,
                BEST_CHECKPOINT
            )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        with open(
            HISTORY_FILE,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )


        print(
            "  Latest checkpoint saved."
        )

        if is_best:

            print(
                "  Best checkpoint saved."
            )


    # ========================================================
    # 23. SAVE FINAL MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        FINAL_MODEL
    )


    total_training_time = (
        time.time() -
        training_start
    )


    # ========================================================
    # 24. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Completed epochs: "
        f"{start_epoch} -> {EPOCHS}"
    )

    print(
        f"Final loss: "
        f"{history[-1]['loss']:.6f}"
    )

    print(
        f"Best loss: "
        f"{best_loss:.6f}"
    )

    print(
        "Training time:",
        round(
            total_training_time / 60,
            2
        ),
        "minutes"
    )

    print(
        "\nFinal model:",
        FINAL_MODEL
    )

    print(
        "Best checkpoint:",
        BEST_CHECKPOINT
    )

    print(
        "Latest checkpoint:",
        LATEST_CHECKPOINT
    )

    print(
        "History:",
        HISTORY_FILE
    )


    # ========================================================
    # 25. RESUME INFORMATION
    # ========================================================

    print("\n" + "=" * 70)
    print("RESUME INFORMATION")
    print("=" * 70)

    print(
        f"Last completed epoch: {EPOCHS}"
    )

    print(
        "To continue training:"
    )

    print(
        "1. Increase EPOCHS."
    )

    print(
        "2. Rerun this script."
    )

    print(
        f"3. It will resume from epoch {EPOCHS + 1}."
    )