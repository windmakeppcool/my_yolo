import random
import os
import numpy as np
import cv2
import os.path as osp
import torch

from torch.utils.data import Dataset
import sys
sys.path.append("/home/liangly/my_projects/myYolo/codes")
from datasets.transforms import random_perspective, Albumentations, augment_hsv, letterbox
from utils.general import xyxy2xywhn, xywhn2xyxy


# 数据增强配置
hyp_config = {
    # affine
    "degrees": 0.0,
    "scale": 0.5,
    "perspective": 0.0,
    "shear": 0.0,
    "translate": 0.1,
    # hsv
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    # flip up-down
    "flipud": 0.0,  # image flip up-down (probability)
    # flip right-left
    "fliplr": 0.5  # image flip left-right (probability)
}


class Yolov5Dataset(Dataset):
    def __init__(self, data_path, mosaic=True,
                 img_size=640):
        super().__init__()
        self.image_path = osp.join(data_path, "images", "train")
        self.label_path = osp.join(data_path, "labels", "train")

        self.indices = [i.rstrip(".txt") for i in os.listdir(self.label_path)]
        self.idx = [i for i in range(len(self.indices))]
        self.mosaic = mosaic
        self.image_size = img_size
        self.mosaic_border = [-img_size//2, img_size//2]
        self.center_ratio_range = (0.5, 1.5)
        self.label_combine = []
        self.albumentations = Albumentations(size=img_size)
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img, labels = self.load_mosaic(index)
        img, labels = random_perspective(img,
                                        labels,
                                        degrees=hyp_config["degrees"],
                                        scale=hyp_config["scale"],
                                        perspective=hyp_config["perspective"],
                                        shear=hyp_config["shear"],
                                        translate=hyp_config["translate"],
                                        border=320)    
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
        img, labels = self.albumentations(img, labels)
        augment_hsv(img, hgain=hyp_config["hsv_h"], sgain=hyp_config["hsv_s"], vgain=hyp_config["hsv_v"])
        # Flip up-down
        if random.random() < hyp_config["flipud"]:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1- labels[:, 2]
        # filp left-right
        if random.random() < hyp_config["fliplr"]:
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        nl = len(labels)
        labels_out = torch.zeros((nl, 6)) # 因为标签长度不一致，第一位用于标识是batch第几个
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        img = img.transpose((2, 0, 1))[::-1] # hwc to chw, bgr to rgb
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, labels_out
    
    @staticmethod
    def collate_fn(batch):
        im, label = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(im, 0), torch.cat(label, 0)

    def load_image(self, index):
        img_path = osp.join(self.image_path, self.indices[index] + ".jpg")
        # print(img_path)
        img = cv2.imread(img_path)
        return img 

    def load_annotation(self, index):
        label_path = osp.join(self.label_path, self.indices[index]+'.txt')
        # print(label_path)
        annotations = np.zeros((0, 5))
        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.rstrip('\n').split(' ')
                annotation = np.zeros((1, 5))
                annotation[0, :] = list(map(float, line))
                annotations = np.append(annotations, annotation, axis=0)
        # convert cx, cy, w, h -> x1, y1, x2, y2
        annotations[:, 1:3] -= annotations[:, 3:] / 2
        annotations[:, 3:] += annotations[:, 1:3]
        return annotations

    def load_mosaic(self, index):
        # 随机选3个图片的index
        indices = [index] + random.choices(self.idx, k=3)
        random.shuffle(indices)
        # 随机生成mosaic
        s = self.image_size
        
        # 遍历四张图片
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        cx = int(random.uniform(*self.center_ratio_range) * s)
        cy = int(random.uniform(*self.center_ratio_range) * s)
        # labels
        labels4 = []
        for i, index in enumerate(indices):
            img = self.load_image(index)
            # resize
            img = self.resize(img)
            h, w, _ = img.shape
            if i == 0:  # 左上
                x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, s * 2), cy
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(s * 2, cy + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = cx, cy, min(cx + w, s * 2), min(cy + h, s * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            padw, padh = x1a - x1b, y1a - y1b
            # 根据图片缩放标签
            anno = self.load_annotation(index)
            anno[:, 1] = (w * anno[:, 1] + padw)
            anno[:, 2] = (h * anno[:, 2] + padh)
            anno[:, 3] = (w * anno[:, 3] + padw)
            anno[:, 4] = (h * anno[:, 4] + padh)
            labels4.append(anno)
        labels4 = np.concatenate(labels4, 0)    # 
        np.clip(labels4, 0, 2 * s, out=labels4)

        # 数据增强
        return img4, labels4

    def resize(self, img):
        h_i, w_i, _ = img.shape
        scale_ration_i = min(self.image_size / h_i, self.image_size / w_i)
        img_i = cv2.resize(img, (int(w_i * scale_ration_i), int(h_i * scale_ration_i)))
        # TODO 随机选择上采样方式
        return img_i


class YoloV5DatasetVal(Yolov5Dataset):
    def __init__(self, data_path, img_size=640):
        self.image_path = osp.join(data_path, "images", "val")
        self.label_path = osp.join(data_path, "labels", "val")
        self.indices = [i.rstrip(".txt") for i in os.listdir(self.label_path)]
        self.idx = [i for i in range(len(self.indices))]
        self.image_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_path = osp.join(self.image_path, self.indices[index]+".jpg")        
        img_ori = cv2.imread(img_path)

        h_ori, w_ori = img_ori.shape[:2]
        label_path = osp.join(self.label_path, self.indices[index]+".txt")
        # cx,cy,w,h, [0,1]
        annotations = np.zeros((0, 5))
        with open(label_path, "r") as f:
            for line in f.readlines():
                line = line.rstrip('\n').split(' ')
                annotation = np.zeros((1, 5))
                annotation[0, :] = list(map(float, line))
                annotations = np.append(annotations, annotation, axis=0)
        
        img, r, pad = self.letterbox(img_ori, self.image_size)
        if len(annotations):
            annotations[:, 1:] = xywhn2xyxy(annotations[:, 1:], r * w_ori, r * h_ori, padh=pad[0], padw=pad[1])
            annotations[:, 1:] = xyxy2xywhn(annotations[:, 1:], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
        
        img = img.transpose((2, 0, 1))[::-1] # hwc to chw, bgr to rgb
        img = np.ascontiguousarray(img)
        data = {
            "img":img, "labels": annotations, "pad":pad, "ori_shape":(h_ori, w_ori)
        }

        return data

    @staticmethod
    def letterbox(img, img_size=640, color=(114, 114, 114), stride=32):
        h_ori, w_ori = img.shape[:2]
        r = min(img_size / h_ori, img_size / w_ori)
        r = min(1.0, r) # 如果边长超过img size则不进行缩放
        h, w = int(r * h_ori), int(r * w_ori)
        if r != 1.0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 补足32位倍数
        pad_h, pad_w = np.mod(h, stride), np.mod(w, stride)
        pad_h = stride - pad_h if pad_h != 0 else pad_h
        pad_w = stride - pad_w if pad_w != 0 else pad_w

        pad_h /= 2
        pad_w /= 2
        
        # 方式pad是奇数导致
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img, r, (pad_h, pad_w)


if __name__ == "__main__":
    data_path = '/home/liangly/datasets/yolov5'
    # dataset = Yolov5Dataset(data_path, is_train=False)
    # for i, data in enumerate(dataset):
    #     img, labels = data
    #     h, w, _ = img.shape
    #     for label in labels:
    #         cx, cy, lw, lh = label[1]*w, label[2]*h, label[3]*w, label[4]*h
    #         x1, y1 = max(0, cx - lw//2), max(0, cy - lh // 2)
    #         x2, y2 = min(cx + lw // 2, w), min(cy + lh // 2, h)
    #         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1, 8)
    #     cv2.imwrite(str(i)+".jpg", img)
    #     if (i == 4):
    #         break

    dataset = YoloV5DatasetVal(data_path)
    for i, data in enumerate(dataset):
        img, labels, pad = data['img'], data['labels'], data['pad']
        img = img.transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[0], img.shape[1]
        if (h % 32 != 0) or (w % 32 != 0):
            print(h, w)
        for label in labels:
            cx, cy, lw, lh = label[1]*w, label[2]*h, label[3]*w, label[4]*h
            x1, y1 = max(0, cx - lw//2), max(0, cy - lh // 2)
            x2, y2 = min(cx + lw // 2, w), min(cy + lh // 2, h)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1, 8)

        cv2.imwrite(str(i) + ".jpg", img)
        if (i == 4):
            break