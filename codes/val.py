import torch

from models.yolov5_model import Yolov5Model
from datasets.yolov5_dataset import YoloV5DatasetVal
from utils.log import get_logger

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def val():
    device = torch.device('cuda')
    logger = get_logger("tmp.log")
    model = Yolov5Model().to(device).eval()
    # load models
    model_dict = torch.load(r'/home/liangly/my_projects/myYolo/work_dir/exp/40.pth')
    model.load_state_dict(model_dict)
    
    path = '/home/liangly/datasets/yolov5'
    dataset = YoloV5DatasetVal(path)

    for i, data in enumerate(dataset):
        img, labels, pad = data['img'], data['labels'], data['pad']
        img = torch.from_numpy(img).to(device).unsqueeze(0) / 255.0
        labels = torch.from_numpy(labels).to(device).unsqueeze(0)

        pred, train_out = model(img)
        # nms



if __name__ == '__main__':
    val()