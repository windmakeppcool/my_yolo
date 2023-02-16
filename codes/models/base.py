import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self,
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                padding=None,
                groups=1,
                dilation=1,
                act=True
                ):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False
            )
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class CSP3Conv(nn.Module):
    # # CSP Bottleneck with 3 convolutions
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        c_middle = int(out_ch * expansion)
        self.cv1 = Conv(in_ch, c_middle, 1, 1)
        self.cv2 = Conv(in_ch, c_middle, 1, 1)
        self.cv3 = Conv(2 * c_middle, out_ch, 1)
        self.m = nn.Sequential(*(Bottleneck(c_middle, c_middle, shortcut, groups, e=1.0)
            for _ in range(n)))
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        c_mid = in_ch // 2
        self.conv_1 = Conv(in_ch, c_mid, 1, 1)
        self.conv_2 = Conv(c_mid * 4, out_ch, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel//2)

    def forward(self, x):
        x = self.conv_1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv_2(torch.cat((x, y1, y2, self.m(y2)), 1))

