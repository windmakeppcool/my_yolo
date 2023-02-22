import torch
import numpy as np
import os
import os.path as osp
from datetime import datetime

from datasets.dataloader import create_dataloader
from models.yolov5_model import Yolov5Model
from loss import ComputeLoss
from utils.log import get_logger
from utils.path import mkdir

anchors=[
    [10,13, 16,30, 33,23],
    [30,61, 62,45, 59,119],
    [116,90, 156,198, 373,326]
]


cfg = {
    "name": "yolov5",
    "root_path": "/home/liangly/my_projects/myYolo/work_dir",
}

def train():
    cfg['name'] = cfg['name'] + '_' + datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    work_path = osp.join(cfg["root_path"], cfg["name"])
    mkdir(work_path)
    # config
    device = torch.device("cuda")
    batch_size = 64
    # get logger
    logger = get_logger(osp.join(work_path, "log.txt"))
    # dataset
    path = '/home/liangly/datasets/yolov5'
    loader, dataset = create_dataloader(path, 640, batch_size)
    # model
    model = Yolov5Model().cuda()
    # loss
    compute_loss = ComputeLoss(anchors=model.head.anchors, device=device)
    # config
    epochs = 300
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(epochs):
        optimizer.zero_grad()
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs = imgs.to(device, non_blocking=True).float() / 255     # 
            pred = model(imgs)
            loss, loss_item = compute_loss(pred, labels.to(device))
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if (i % 100 == 0):
                lbox, lobj, lcls = loss_item[0].item(), loss_item[1].item(), loss_item[2].item()
                logger.info("Epoch {}, iter {}, \
                    box loss: {:.4f}, obj loss {:.4f}, cls loss {:.4f}, total loss {:.4f}".format(
                        epoch, i, lbox, lobj, lcls, loss.item() / batch_size))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "/home/liangly/my_projects/myYolo/work_dir/exp/epoch_{}.pth".format(epoch+1))

        scheduler.step()


if __name__ == "__main__":
    train()