import torch
import torch.nn as nn


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def conv_block(in_c, out_c):

    mid = out_c // 2

    return nn.Sequential(

        nn.Conv2d(
            in_c,
            mid,
            1
        ),

        nn.BatchNorm2d(
            mid
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Conv2d(
            mid,
            mid,
            3,
            padding=1,
            groups=mid
        ),

        nn.BatchNorm2d(
            mid
        ),

        nn.Conv2d(
            mid,
            out_c,
            1
        ),

        nn.BatchNorm2d(
            out_c
        ),

        nn.ReLU(
            inplace=True
        )
    )


class ChannelAttention(nn.Module):

    def __init__(
        self,
        c,
        reduction=16
    ):

        super().__init__()

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.max_pool = (
            nn.AdaptiveMaxPool2d(1)
        )

        hidden = max(
            c // reduction,
            1
        )

        self.mlp = nn.Sequential(

            nn.Conv2d(
                c,
                hidden,
                1,
                bias=False
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                hidden,
                c,
                1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        avg_out = self.mlp(
            self.avg_pool(x)
        )

        max_out = self.mlp(
            self.max_pool(x)
        )

        attention = self.sigmoid(
            avg_out +
            max_out
        )

        return x * attention


class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()

        padding = (
            kernel_size // 2
        )

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        attention = torch.cat(
            [
                avg_out,
                max_out
            ],
            dim=1
        )

        attention = self.sigmoid(
            self.conv(attention)
        )

        return x * attention


class DualAttention(nn.Module):

    def __init__(self, c):

        super().__init__()

        self.ca = ChannelAttention(c)

        self.sa = SpatialAttention()


    def forward(self, x):

        x = self.ca(x)

        x = self.sa(x)

        return x


class UNetDual(nn.Module):

    def __init__(self):

        super().__init__()

        # ENCODER

        self.enc1 = conv_block(
            3,
            64
        )

        self.enc2 = conv_block(
            64,
            128
        )

        self.enc3 = conv_block(
            128,
            256
        )

        self.enc4 = conv_block(
            256,
            512
        )

        self.pool = nn.MaxPool2d(
            2
        )

        # BOTTLENECK ATTENTION

        self.attn = DualAttention(
            512
        )

        # DECODER

        self.up1 = nn.ConvTranspose2d(
            512,
            256,
            2,
            2
        )

        self.dec1 = conv_block(
            256,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )

        self.dec2 = conv_block(
            128,
            128
        )

        self.up3 = nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )

        self.dec3 = conv_block(
            64,
            64
        )

        self.out = nn.Conv2d(
            64,
            3,
            1
        )


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        e4 = self.attn(e4)

        d1 = self.up1(e4)

        d1 = d1 + e3

        d1 = self.dec1(d1)

        d2 = self.up2(d1)

        d2 = d2 + e2

        d2 = self.dec2(d2)

        d3 = self.up3(d2)

        d3 = d3 + e1

        d3 = self.dec3(d3)

        return torch.sigmoid(
            self.out(d3)
        )