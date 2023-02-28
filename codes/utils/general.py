import torch
import numpy as np
import torchvision


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def scale_boxes(src_shape, dst_shape, boxes):
    gain = min(src_shape[0] / dst_shape[0], src_shape[1] / dst_shape[1]) # gain = old / new
    pad = (src_shape[1] - dst_shape[1] * gain) / 2, (src_shape[0] * gain) / 2

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    
    clip_boxes(boxes, dst_shape)
    return boxes


def nms(pred, conf_thres=0.25, iou_thres=0.45, multi_labels=False, agnostic=False, max_det=300):
    max_wh = 7680   # 这里不同类别的nms单独做nms,所以根据idx对不同类的框施加一个偏移量
    max_nms = 30000 # 最大框数量
    
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    pred = pred.cpu()
    bs = pred.shape[0]
    nc = pred.shape[2] - 5
    xc = pred[..., 4] > conf_thres

    mi = 5 + nc # mask start index
    output = [torch.zeros((0, 6))] * bs
    for i in range(bs):
        x = pred[i][xc[i]]
        #compute conf
        x[:, 5:] *= x[:, 4:5]
        # Box/Mask
        box = xywh2xyxy(x[..., :4]) # cx,cy,w,h -> x1,y1,x2,y2
        
        if multi_labels:
            #TODO
            pass
        else:
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]# [..., 6]

        n_box = x.shape[0]
        if not n_box:
            continue
        # 按照置信度降序，且只取前最大个数
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        # batch nms
        offset = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + offset, x[:, 4]
        nms_i = torchvision.ops.nms(boxes, scores, iou_thres)[:max_det]

        output[i] = x[nms_i]
    # x1y1x2y2,conf,label
    return output