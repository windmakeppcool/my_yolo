import torch

from models.yolov5_model import Yolov5Model
from datasets.yolov5_dataset import YoloV5DatasetVal
from utils.log import get_logger
from utils.general import nms, scale_boxes
from utils.plots import plot_images, output_to_targets

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"


def val():
    device = torch.device('cuda')
    logger = get_logger("tmp.log")
    model = Yolov5Model().to(device).eval()
    # load models
    model_dict = torch.load(r'/home/liangly/my_projects/myYolo/work_dir/exp/300.pth')
    model.load_state_dict(model_dict)
    
    path = '/home/liangly/datasets/yolov5'
    dataset = YoloV5DatasetVal(path)

    for i, data in enumerate(dataset):
        img, labels, ori_shape = data['img'], data['labels'], data['ori_shape']
        img = torch.from_numpy(img).to(device).unsqueeze(0) / 255.0
        labels = torch.from_numpy(labels).to(device)#.unsqueeze(0)
        with torch.no_grad():
            preds, train_out = model(img)
        # nms
        preds = nms(preds, 0.25, 0.6)
        # # metrics
        for si, pred in enumerate(preds):
            num_label, num_pred = labels.shape[0], pred.shape[0]

            # pred[:, 5] = 0
            # predn = pred.clone()
            # scale_boxes(img[0].shape[1:], ori_shape, predn[:, :4])

        if i == 33:
            # plot_images(img, labels, "tmp_{}.jpg".format(i))
            plot_images(img, output_to_targets(pred), "tmp_pred_{}.jpg".format(i))
            break


if __name__ == '__main__':
    val()