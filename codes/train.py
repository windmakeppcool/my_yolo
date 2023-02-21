import torch
import numpy as np

from datasets.dataloader import create_dataloader
from models.yolov5_model import Yolov5Model
from loss import ComputeLoss

anchors=[
    [10,13, 16,30, 33,23],
    [30,61, 62,45, 59,119],
    [116,90, 156,198, 373,326]
]

def train():
    device = torch.device("cuda")
    batch_size = 64
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
                print("Epoch {}, iter {}, \
                    box loss: {:.4f}, obj loss {:.4f}, cls loss {:.4f}, total loss {:.4f}".format(
                        epoch, i, lbox, lobj, lcls, loss.item() / batch_size))

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), "/home/liangly/my_projects/myYolo/work_dir/exp/{}.pth".format(epoch+1))

        scheduler.step()


if __name__ == "__main__":
    train()