
import os.path as osp
import cv2
import numpy as np


from torch.utils.data import Dataset
from pycocotools.coco import COCO


METAINFO = {
    'CLASSES':
    ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
    # PALETTE is a list of color tuples, which is used for visualization.
    'PALETTE':
    [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
        (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
        (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
        (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
        (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
        (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
        (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
        (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
        (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
        (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
        (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
        (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
        (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
        (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
        (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
        (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
        (246, 0, 122), (191, 162, 208)]
}
# 数据格式
# 'images': [
#     {
#         'file_name': 'COCO_val2014_000000001268.jpg',
#         'height': 427,
#         'width': 640,
#         'id': 1268
#     },
#     ...
# ],

# 'annotations': [
#     {
#         'segmentation': [[192.81,
#             247.09,
#             ...
#             219.03,
#             249.06]],  # if you have mask labels
#         'area': 1035.749,
#         'iscrowd': 0,
#         'image_id': 1268,
#         'bbox': [192.81, 224.8, 74.73, 33.43],
#         'category_id': 16,
#         'id': 42986
#     },
#     ...
# ],

# 'categories': [
#     {'id': 0, 'name': 'car'},
#      ...
#  ]

class LoadCocoDataset(Dataset):
    def __init__(self, data_path, ann_file):
        super().__init__()
        self.data_path = data_path
        self.ann_file = ann_file

        self.coco_dataset = COCO(self.ann_file)
        # image_ids = [397133, 37777, 252219, 87038, 174482, 403385, 6818, 480985, 458054,...]
        self.image_ids = self.coco_dataset.getImgIds()
        self.indices = [i for i in range(len(self.image_ids))]
        self.load_label()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        anno = self.load_annotation(index)
        sample = {'img': img, 'anno': anno}
        return sample
    
    def load_label(self):
        # from name to label
        categories = self.coco_dataset.loadCats(self.coco_dataset.getCatIds())
        categories.sort(key=lambda x: x["id"])
        self.classes = {} # class name to label
        self.labels = {} # label to class name
        self.coco_labels_to_id = {}
        self.coco_id_to_label = {}

        # 这里categories中id不连续且不从0开始，所以采用len的方式更新id
        for c in categories:
            self.coco_labels_to_id[len(self.classes)] = c['id']
            self.coco_id_to_label[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)
            
        for key, value in self.classes.items():
            self.labels[value] = key

    def load_image(self, image_index):
        image_info = self.coco_dataset.loadImgs(self.image_ids[image_index])[0]
        img_path = osp.join(self.data_path, image_info['file_name'])
        img = cv2.imread(img_path) # bgr
        return img

    def load_annotation(self, image_index):
        annotations_ids = self.coco_dataset.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations
        
        coco_annotations = self.coco_dataset.loadAnns(annotations_ids)
        # coco anno is x,y,w,h, we need x1,y1,x2,y2
        for _, anno in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if anno['bbox'][2] < 1 or anno['bbox'][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = anno['bbox']
            annotation[0, 4] = self.coco_id_to_label[anno['category_id']]
            annotations = np.append(annotations, annotation, axis=0)
        # convert xywh ->x1y1x2y2
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        # 一张图片里有许多标注
        return annotations  # x1, y1, w, h, label




    


if __name__ == "__main__":
    data_path = r'/home/liangly/datasets/coco/val2017'
    ann_path = r'/home/liangly/datasets/coco/annotations/instances_val2017.json'

    dataset = LoadCocoDataset(data_path, ann_path)

    for i, data in enumerate(dataset):
        img, labels = data['img'], data['anno']
        for label in labels:
            cv2.rectangle(img, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), (0, 0, 255), 1, 8)
        cv2.imwrite(str(i)+".jpg", img)
        break
