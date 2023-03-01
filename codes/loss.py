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

        self.BCEcls, self.BCEobj = BCEcls, BCEobj

        self.num_of_anchors = 3
        self.num_of_class = 80
        self.num_of_outputs = 3
        self.device = device
        self.anchors = anchors  # 被stride归一化后的
        self.anchor_t = 4.0

    def __call__(self, preds, targets):
        lcls = torch.zeros(1, device=self.device)   # class loss
        lbox = torch.zeros(1, device=self.device)   # box loss
        lobj = torch.zeros(1, device=self.device)   # object loss
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)

        for i, pred_i in enumerate(preds):
            batch_idx, anchor_idx, grid_j, grid_i = indices[i]
            tobj = torch.zeros(pred_i.shape[:4], dtype=pred_i.dtype, device=self.device)
            n = batch_idx.shape[0]
            if n:
                # target-subset of predictions
                pxy, pwh, _, pcls = pred_i[batch_idx, anchor_idx, grid_j, grid_i].split((2, 2, 1, self.num_of_class), 1)
                # regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()# iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                # object
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[batch_idx, anchor_idx, grid_j, grid_i] = iou
                # classification
                if self.num_of_class > 1:
                    t = torch.full_like(pcls, 0.0, device=self.device)
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.BCEcls(pcls, t)
            
            obji = self.BCEobj(pred_i[..., 4], tobj)
            lobj += obji * self.balance[i]
        
        lbox *= hyp_config["box"]
        lobj *= hyp_config["obj"]
        lcls *= hyp_config["cls"]
        bs = tobj.shape[0]

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, preds, targets):
        indices, tbox, anch, tcls = [], [], [], []
        # target [num,[image,class,x,y,w,h]]
        num_of_targets = targets.shape[0]
        anchor_idx = torch.arange(self.num_of_anchors, device=self.device).float()
        anchor_idx = anchor_idx.view(self.num_of_anchors, 1).repeat(1, num_of_targets)
        targets = torch.cat((targets.repeat(self.num_of_anchors, 1, 1), anchor_idx[..., None]), 2)
        # targets [num_of_anchors, num_targets, [image, class, xywh, anchor idx]]
        # gain 
        gain = torch.ones(7, device=self.device)

        g = 0.5
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.num_of_outputs):
            anchors, shape = self.anchors[i], preds[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            t = targets * gain # 归一化到输出特征图大小
            if num_of_targets:
                wh_ratio = t[..., 4:6] / anchors[:, None] # wh ratio
                # 根据比例排除差距过大的框,ratio_bool->shape(3,num_of_targes)
                ratio_bool = torch.max(wh_ratio, 1 / wh_ratio).max(2)[0] < self.anchor_t
                t = t[ratio_bool]   # shape: (3,num_of_targets,7)->(num after filter, 7)

                #offsets
                grid_xy = t[:, 2:4]
                grid_xy_inverse = gain[[2, 3]] - grid_xy
                left, top = ((grid_xy % 1 < g) & (grid_xy > 1.)).T
                right, bottom = ((grid_xy_inverse % 1 < g) & (grid_xy_inverse > 1.)).T
                # shape of (left, top, right, bottom) is (num_after_filter)
                all_offsets = torch.stack((torch.ones_like(left), left, top, right, bottom))
                t = t.repeat((5, 1, 1))[all_offsets]
                offsets = (torch.zeros_like(grid_xy)[None] + off[:, None])[all_offsets]
            else:
                t = targets[0]
                all_offsets = 0
            # define
            batch_and_class, grid_xy, grid_wh, anchor_idx = t.chunk(4, 1)
            anchor_idx = anchor_idx.long().view(-1)
            (batch, classes) = batch_and_class.long().T
            grid_ij = (grid_xy - offsets).long()
            grid_i, grid_j = grid_ij.T

            # append
            indices.append((batch, anchor_idx, grid_j.clamp_(0, shape[2]-1), grid_i.clamp_(0, shape[3]-1)))
            tbox.append(torch.cat((grid_xy - grid_ij, grid_wh), 1))
            anch.append(anchors[anchor_idx])
            tcls.append(classes)
        
        return tcls, tbox, indices, anch
            

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

