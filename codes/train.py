import torch
import numpy as np
import os
import os.path as osp
import sys
from datetime import datetime

import wandb

root_path = osp.abspath(osp.dirname(__file__))
root_path = osp.abspath(osp.dirname(root_path))
print("root_path: ", root_path)
sys.path.append(root_path)
sys.path.append(osp.join(root_path, 'codes'))

from datasets.dataloader import create_dataloader
from models.yolov5_model import Yolov5Model
from loss import ComputeLoss
from utils.log import get_logger
from utils.path import mkdir

os.environ['CUDA_VISIBLE_DEVICES'] = "2"


anchors=[
    [10,13, 16,30, 33,23],
    [30,61, 62,45, 59,119],
    [116,90, 156,198, 373,326]
]


cfg = {
    "name": "yolov5s",
    "root_path": osp.join(root_path, "wandb"),
    "dataset_path": "/home/liangly/datasets/yolov5",
    "epochs": 150,
    "batch_size": 64,
    "lr": 0.001
}

wandb.init(
    project='yolov5',
    config=cfg,
    dir=cfg["root_path"]
)


def train():
    cfg['name'] = cfg['name'] + '_' + datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    work_path = osp.join(cfg["root_path"], cfg["name"])
    mkdir(work_path)
    # config
    device = torch.device("cuda")
    batch_size = wandb.config.batch_size
    # get logger
    logger = get_logger(osp.join(work_path, "log.txt"))
    # dataset
    loader, dataset = create_dataloader(cfg["dataset_path"], img_size=640, batch_size=batch_size)
    # model
    model = Yolov5Model().cuda()
    # loss
    compute_loss = ComputeLoss(anchors=model.head.anchors, device=device)
    # config
    epochs = wandb.config.epochs
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=wandb.config.lr, momentum=0.9, nesterov=True)
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
                lbox = loss_item[0].item() * batch_size
                lobj = loss_item[1].item() * batch_size
                lcls = loss_item[2].item() * batch_size

                logger.info("Epoch {}, iter {}, \
                    box loss: {:.4f}, obj loss {:.4f}, cls loss {:.4f}, total loss {:.4f}".format(
                        epoch, i, lbox, lobj, lcls, loss.item() / batch_size))
                wandb.log({"obj loss":lobj, "box loss":lbox, "cls loss": lcls, "total loss": loss.item()})
        
            
        if (epoch + 1) % 10 == 0:
            # torch.save(model.state_dict(), "/home/liangly/my_projects/myYolo/work_dir/exp/epoch_{}.pth".format(epoch+1))
            torch.save(model.state_dict(), osp.join(work_path, 'epoch_{}.pth'.format(epoch+1)))
        wandb.log({"learning rate": optimizer.state_dict()['param_groups'][0]['lr']})
        scheduler.step()


if __name__ == "__main__":
    train()