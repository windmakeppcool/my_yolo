import random
import os
import numpy as np
import cv2

from torch.utils.data import Dataset


class Yolov5Dataset(Dataset):
    def __init__(self, data_path, ann_file, mosaic=True,
                 img_size=640):
        super().__init__(data_path, ann_file)
        self.mosaic = mosaic
        self.image_size = img_size
        self.mosaic_border = [-img_size//2, img_size//2]
        self.center_ratio_range = (0.5, 1.5)
        self.label_combine = []

    def __getitem__(self, index):
        if not self.mosaic:
            return super().__getitem__(index)
        img, labels = self.load_mosaic(index)

        return img, labels

    def load_mosaic(self, index):
        # 随机选3个图片的index
        indices = [index] + random.choices(self.indices, k=3)
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
            anno[:, 0] = (w * anno[:, 0] + padw)
            anno[:, 1] = (h * anno[:, 1] + padh)
            anno[:, 2] = (w * anno[:, 2] + padw)
            anno[:, 3] = (h * anno[:, 3] + padh)
            labels4.append(anno)
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4, 0, 2 * s, out=labels4)

        return img4, labels4

    def resize(self, img):
        h_i, w_i, _ = img.shape
        scale_ration_i = min(self.image_size / h_i, self.image_size / w_i)
        img_i = cv2.resize(img, (int(w_i * scale_ration_i), int(h_i * scale_ration_i)))
        # TODO 随机选择上采样方式
        return img_i