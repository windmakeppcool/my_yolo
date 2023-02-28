import torch
import numpy as np
import cv2

from utils.general import xywh2xyxy
from datasets.coco_dataset import METAINFO

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


class Annotator:
    def __init__(self, image, line_width=None, font_size=None):
        assert image.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.image = image
        self.lw = line_width or max(round(sum(image.shape) / 2 * 0.003), 2) # line width
    
    def box_label(self, box, label='', color=(128,128,128), txt_color=(255,255,255)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.image, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)
            # text width, height
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.image, p1, p2, color, -1, cv2.LINE_AA) # filled
            cv2.putText(self.image,
                        label,
                        (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        self.lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)
    


def plot_images(images, targets, fname="", is_label=True):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().float().detach().numpy()


    if np.max(images[0]) <= 1:
        images *= 255  # de-normalise (optional)
    images = images[0,...].transpose((1, 2, 0))[:,:,::-1]
    h, w = images.shape[:2]
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = np.ascontiguousarray(images)
    annotator = Annotator(images)
    if len(targets) > 0:
        if is_label:
            boxes = xywh2xyxy(targets[:, 1:5]).T
        else:
            boxes = targets[:, 1:5]
            boxes = boxes.T
        classes = targets[:, 0].astype('int')
        if boxes.shape[1]:
            # if boxes.max() <= 1.01:
            if is_label:
                boxes[[0, 2]] *= w
                boxes[[1, 3]] *= h
        
        for j, box in enumerate(boxes.T.tolist()):
            cls = classes[j]
            color = METAINFO['PALETTE'][cls]
            name = METAINFO['CLASSES'][cls]

            annotator.box_label(box, name, color)

    if fname != "":
        cv2.imwrite(fname, images)


def output_to_targets(output):
    box, conf, cls = output[:, :6].cpu().split((4, 1, 1), 1)
    targets = torch.cat((cls, box, conf), 1)
    return targets
