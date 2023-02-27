import torch.nn as nn
import torch

from utils.metrics import bbox_iou


anchors=[
    [10,13, 16,30, 33,23],
    [30,61, 62,45, 59,119],
    [116,90, 156,198, 373,326]
]

hyp_config = {
    "box": 0.05,
    "obj": 1.0,
    "cls": 0.5
}

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class ComputeLoss:
    def __init__(self, anchors, device):
        self.balance = [4.0, 1,0, 0.4] # nl=3, 不同层输出对应的系数
        self.stride = [8, 16, 32] # 不同层缩放的倍数

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=0.0) # positive, negative BCE targets

        self.BCEcls, self.BCEobj = BCEcls, BCEobj

        self.na = 3
        self.nc = 80
        self.nl =3
        self.device = device
        self.anchors = anchors
        self.anchor_t = 4.0
        self.g = 0.5    # 偏移量
        self.off = torch.tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ], device=device).float() * self.g # shape(5, 2)

    def __call__(self, pred, targets):
        lcls = torch.zeros(1, device=self.device)   # class loss
        lbox = torch.zeros(1, device=self.device)   # box loss
        lobj = torch.zeros(1, device=self.device)   # object loss
        tcls, tbox, indices, anchors = self.build_targets(pred, targets)

        # 计算loss
        for i, pred_i in enumerate(pred):
            b, anch, grid_j, grid_i = indices[i] # image, anchor, gridy, gridx
            tobj = torch.zeros(pred_i.shape[:4], dtype=pred_i.dtype, device=self.device)
            n = b.shape[0] # number of targets
            if n:
                pxy, pwh, _, pcls = pred_i[b, anch, grid_j, grid_i].split((2, 2, 1, self.nc), 1)
                # regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean() # iou loss

                # objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[b, anch, grid_j, grid_i] = iou
                # classification
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)    # BCE

            obji = self.BCEobj(pred_i[..., 4], tobj)
            lobj += obji * self.balance[i]

        lbox *= hyp_config["box"]
        lobj *= hyp_config["obj"]
        lcls *= hyp_config["cls"]
        bs = tobj.shape[0]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, pred, targets):
        # target [image, class, x,y,w,h]
        na, nt = self.na, targets.shape[0] # num of anchors, img index
        tcls, tbox, indices, anchors_list = [], [], [], []
        gain = torch.ones(7, device=self.device)
        # 复制gt box，anchor有三种类型，每种都与gt box与之对应
        # 给这些框加上索引，每个对应是哪种类型的框
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        for i in range(self.nl):
            anchor, shape = self.anchors[i], pred[i].shape
            anchor = torch.div(anchor, self.stride[i])
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]] # xyxy gain
            # 将标注投射到特征图大小上
            t = targets * gain # shape(3, n, 7)
            if nt:
                ratio = t[..., 4:6] / anchor[:, None] # wh ration
                # bool array,shape(3, n)
                ''' 过滤掉gt box与anchor较大的部分, 该部分不易训练'''
                judge_true = torch.max(ratio, 1. / ratio).max(2)[0] < self.anchor_t
                t = t[judge_true]   # (3, n, 7) -> (m, 7)

                # offsets
                grid_xy = t[:, 2:4] # grid xy, 左上角为0点
                grid_xy_inverse = gain[[2, 3]] - grid_xy # inverse 右下角为0点
                j, k = ((grid_xy % 1 < self.g) & (grid_xy > 1)).T # shape (m), (m)
                l, m = ((grid_xy_inverse % 1 < self.g) & (grid_xy_inverse > 1)).T   # shape (m), (m)
                j = torch.stack((torch.ones_like(j), j, k, l, m))   #shape (5, m)
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(grid_xy)[None] + self.off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # define
            # 7->2,2,2,1 (img_idx, class), xy, wh, anchor_idx
            bc, grid_xy, grid_wh, anch = t.chunk(4, 1)
            anch, (b, c) = anch.long().view(-1), bc.long().T   # 
            grid_ij = (grid_xy - offsets).long()
            grid_i, grid_j = grid_ij.T

            # append
            indices.append((b, anch, grid_j.clamp_(0, shape[2] - 1), grid_i.clamp_(0, shape[3] - 1))) # image, anchor, grid
            tbox.append(torch.cat((grid_xy - grid_ij, grid_wh), 1)) # box, 预测的是偏移量
            anchors_list.append(anchor[anch]) # anchors
            tcls.append(c)  # class 

        return tcls, tbox, indices, anchors_list


if __name__ == "__main__":
    import cv2
    from pycocotools.coco import COCO
    coco_anno_path = r'/home/liangly/datasets/coco/annotations/instances_val2017.json'
    img_path = r'/home/liangly/datasets/coco'
    coco_dataset = COCO(annotation_file=coco_anno_path)
    image_idx = coco_dataset.getImgIds()[0]
    image_info = coco_dataset.loadImgs(image_idx)[0]
    anno_idx = coco_dataset.getAnnIds(image_idx, iscrowd=False)
    coco_anno = coco_dataset.loadAnns(anno_idx)[0]
    x, y, w, h = map(int, coco_anno["bbox"])
    x1, y1, x2, y2 = x, y, x + w, y + h
    img = cv2.imread(r"/home/liangly/datasets/coco/val2017/" + image_info["file_name"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1, 8)
    # cv2.imwrite("tmp.jpg", img)
    # img to (640, 640)
    height, width = img.shape[:2]
    ratio = min(640 / height, 640 / width)
    # cv2.resize()

