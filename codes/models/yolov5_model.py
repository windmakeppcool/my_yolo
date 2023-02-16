import sys
import platform
import os
import torch
import torch.nn as nn

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root dir

if str(ROOT) not in sys.path:   
    sys.path.append(str(ROOT))
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from base import Conv, CSP3Conv, SPPF


yolov5s_cfg = {
    # num_conv : [input_ch, out_ch, kernel, stride]
    0 : [3, 32, 6, 2],
    1 : [32, 64, 3, 2],
    2 : [64, 128, 3, 2],
    3 : [128, 256, 3, 2],
    4 : [256, 512, 3, 2],
}


class Stage(nn.Module):
    c3_stage = [1, 2, 3, 4]
    def __init__(self,
                stage,
                num_layers,
                input_ch,
                output_ch,
                kernel_size,
                stride,
                padding=None):
        super().__init__()
        stage_list = []
        stage_list.append(Conv(input_ch, output_ch, kernel_size, stride, padding))

        if stage in self.c3_stage:
            stage_list.append(CSP3Conv(output_ch, output_ch, num_layers))
        self.stage = nn.Sequential(*stage_list)

    def forward(self, x):
        x = self.stage(x)
        return x


class YoloV5Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_1 = Stage(0, 1, *yolov5s_cfg[0], 2)
        self.stage_2 = Stage(1, 3, *yolov5s_cfg[1])
        self.stage_3 = Stage(2, 6, *yolov5s_cfg[2])
        self.stage_4 = Stage(3, 9, *yolov5s_cfg[3])
        self.stage_5 = Stage(4, 3, *yolov5s_cfg[4])
        
        self.sppf = SPPF(yolov5s_cfg[4][1],
                         yolov5s_cfg[4][1], 5)

    def forward(self, x):
        out_1 = self.stage_1(x)
        out_2 = self.stage_2(out_1)
        out_3 = self.stage_3(out_2)
        out_4 = self.stage_4(out_3)
        out_5 = self.stage_5(out_4)
        out_5 = self.sppf(out_5)

        return out_3, out_4, out_5


class YoloV5Neck(nn.Module):
    def __init__(self):
        super().__init__()
        out_channels = []
        for i in range(2, 5):
            out_channels.append(yolov5s_cfg[i][1])

        assert(len(out_channels) == 3)

        self.cv1 = Conv(out_channels[2], out_channels[1], 1)
        self.cv2 = Conv(out_channels[1], out_channels[0], 1)
        self.cv3 = Conv(out_channels[0], out_channels[0], 3, 2)
        self.cv4 = Conv(out_channels[1], out_channels[1], 3, 2)

        self.c3_1 = CSP3Conv(out_channels[2], out_channels[1], 3, False)
        self.c3_2 = CSP3Conv(out_channels[1], out_channels[0], 3, False)
        self.c3_3 = CSP3Conv(out_channels[1], out_channels[1], 3, False)
        self.c3_4 = CSP3Conv(out_channels[2], out_channels[2], 3, False)

        self.up_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, out_1, out_2, out_3):
        x1 = self.cv1(out_3)
        out_3 = self.up_1(x1)
        out_2 = torch.cat([out_3, out_2], 1)
        out_2 = self.c3_1(out_2)
        x2 = self.cv2(out_2)
        out_2 = self.up_2(x2)
        x3 = torch.cat([out_2, out_1], 1)

        x3 = self.c3_2(x3)
        out_3 = self.cv3(x3)
        x2 = torch.cat([out_3, x2], 1)
        x2 = self.c3_3(x2)
        out_2 = self.cv4(x2)
        x1 = torch.cat([out_2, x1], 1)
        x1 = self.c3_4(x1)

        return x3, x2, x1


class Yolov5Head(nn.Module):
    num_class = 80
    anchors=[
        [10,13, 16,30, 33,23],
        [30,61, 62,45, 59,119],
        [116,90, 156,198, 373,326]
    ]
    export = False
    stride = [8, 16, 32]
    def __init__(self):
        super().__init__()
        ch = []
        for i in range(2, 5):
            ch.append(yolov5s_cfg[i][1])
        
        self.nc = self.num_class
        self.no = self.nc + 5 # number of outputs per anchor
        self.nl = len(self.anchors) # number of detection layers
        self.na = len(self.anchors[0]) // 2 # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.anchors = torch.tensor(self.anchors).view(3, 3, 2).cuda()

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            # x(bs,255,20,20) to x(bs,3,20,20,85)
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))
        
        if self.training:
            return x
        else:
            if self.export:
                return torch.cat(z, 1)
            else:
                return (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        # d, t = torch.cuda, torch.float64
        shape = 1, self.na, ny, nx, 2 # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x) 
         # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)

        return grid, anchor_grid

class Yolov5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = YoloV5Backbone()
        self.neck = YoloV5Neck()
        self.head = Yolov5Head()

    def forward(self, x):
        out_1, out_2, out_3 = self.backbone(x)
        out_1, out_2, out_3 = self.neck(out_1, out_2, out_3)
        out = self.head([out_1, out_2, out_3])

        return out

if __name__ == '__main__':
    model = Yolov5Model().cuda()
    # model.training = True
    model.eval()
    input_tensor = torch.ones((1, 3, 640, 640)).cuda()

    output = model(input_tensor)