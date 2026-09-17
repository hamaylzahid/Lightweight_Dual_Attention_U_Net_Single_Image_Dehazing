from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as T


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


def get_image_files(directory):

    directory = Path(directory)

    return {
        p.name: p
        for p in directory.iterdir()
        if (
            p.is_file()
            and
            p.suffix.lower()
            in VALID_EXTENSIONS
        )
    }


class TestDataset(Dataset):

    def __init__(
        self,
        pairs,
        size=256
    ):

        self.pairs = pairs

        self.transform = T.Compose([

            T.Resize(
                (
                    size,
                    size
                )
            ),

            T.ToTensor()

        ])


    def __len__(self):

        return len(
            self.pairs
        )


    def __getitem__(
        self,
        idx
    ):

        hazy_path, gt_path = (
            self.pairs[idx]
        )

        hazy = Image.open(
            hazy_path
        ).convert("RGB")

        gt = Image.open(
            gt_path
        ).convert("RGB")

        hazy = self.transform(
            hazy
        )

        gt = self.transform(
            gt
        )

        return (
            hazy,
            gt,
            hazy_path.name
        )


def create_test_pairs(
    test_hazy_dir,
    test_gt_dir
):

    test_hazy_files = get_image_files(
        test_hazy_dir
    )

    test_gt_files = get_image_files(
        test_gt_dir
    )

    hazy_names = set(
        test_hazy_files.keys()
    )

    gt_names = set(
        test_gt_files.keys()
    )

    matching_names = (
        hazy_names &
        gt_names
    )

    assert len(test_hazy_files) == 1000
    assert len(test_gt_files) == 1000
    assert len(matching_names) == 1000
    assert len(hazy_names - gt_names) == 0
    assert len(gt_names - hazy_names) == 0

    test_pairs = [

        (
            test_hazy_files[name],
            test_gt_files[name]
        )

        for name in sorted(
            matching_names
        )

    ]

    assert len(test_pairs) == 1000

    return test_pairs


def create_test_loader(
    test_pairs,
    image_size=256,
    batch_size=8,
    num_workers=2
):

    test_dataset = TestDataset(
        test_pairs,
        image_size
    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False

    )

    return test_dataset, test_loader