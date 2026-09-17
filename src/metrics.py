import torch


def gaussian_kernel(
    window_size=11,
    sigma=1.5,
    channels=3
):

    coords = torch.arange(
        window_size,
        dtype=torch.float32
    )

    coords -= (
        window_size - 1
    ) / 2

    gaussian = torch.exp(
        -(
            coords ** 2
        ) /
        (
            2 *
            sigma ** 2
        )
    )

    gaussian /= (
        gaussian.sum()
    )

    kernel_2d = (
        gaussian[:, None]
        *
        gaussian[None, :]
    )

    kernel_2d = (
        kernel_2d
        / kernel_2d.sum()
    )

    kernel = (
        kernel_2d
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(
            channels,
            1,
            1,
            1
        )
    )

    return kernel


def ssim(
    img1,
    img2,
    window_size=11,
    sigma=1.5,
    data_range=1.0
):

    channels = img1.size(1)

    window = gaussian_kernel(
        window_size,
        sigma,
        channels
    ).to(
        img1.device,
        dtype=img1.dtype
    )

    padding = (
        window_size // 2
    )

    mu1 = torch.nn.functional.conv2d(
        img1,
        window,
        padding=padding,
        groups=channels
    )

    mu2 = torch.nn.functional.conv2d(
        img2,
        window,
        padding=padding,
        groups=channels
    )

    mu1_sq = mu1.pow(2)

    mu2_sq = mu2.pow(2)

    mu1_mu2 = (
        mu1 *
        mu2
    )

    sigma1_sq = (
        torch.nn.functional.conv2d(
            img1 * img1,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu1_sq
    )

    sigma2_sq = (
        torch.nn.functional.conv2d(
            img2 * img2,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu2_sq
    )

    sigma12 = (
        torch.nn.functional.conv2d(
            img1 * img2,
            window,
            padding=padding,
            groups=channels
        )
        -
        mu1_mu2
    )

    C1 = (
        0.01 *
        data_range
    ) ** 2

    C2 = (
        0.03 *
        data_range
    ) ** 2

    numerator = (
        (2 * mu1_mu2 + C1)
        *
        (2 * sigma12 + C2)
    )

    denominator = (
        (mu1_sq + mu2_sq + C1)
        *
        (sigma1_sq + sigma2_sq + C2)
    )

    ssim_map = (
        numerator /
        (denominator + 1e-12)
    )

    return ssim_map.mean(
        dim=(1, 2, 3)
    )


def calculate_psnr(
    pred,
    target,
    data_range=1.0
):

    mse = torch.mean(
        (pred - target) ** 2,
        dim=(1, 2, 3)
    )

    psnr = torch.where(

        mse > 0,

        10.0 *
        torch.log10(
            (data_range ** 2) /
            mse
        ),

        torch.full_like(
            mse,
            float("inf")
        )

    )

    return psnr