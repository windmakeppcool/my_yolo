from pycocotools.coco import COCO
import os.path as osp
import os
import shutil
import numpy as np


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25, None,
         None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
         51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
         None, 73, 74, 75, 76, 77, 78, 79, None]
    return x

def mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    root_path = '/home/liangly/datasets/coco'
    output_path = '/home/liangly/datasets/yolov5'
    mkdir(output_path)
    # flag_list = ['train', 'val']
    flag_list = ['train']
    anno_path = osp.join(root_path, 'annotations')

    coco80 = coco91_to_coco80_class()
    for flag in flag_list:
        #mkdir
        output_image_path = osp.join(output_path, "images")
        mkdir(output_image_path)
        output_label_path = osp.join(output_path, "labels")
        mkdir(output_label_path)
        mkdir(osp.join(output_path, "images", flag))
        mkdir(osp.join(output_path, "labels", flag))

        # 
        json_path = osp.join(anno_path, "instances_{}2017.json".format(flag))
        image_root_path = osp.join(root_path, "{}2017".format(flag))
        coco_dataset = COCO(json_path)
        image_ids = coco_dataset.getImgIds()
        for image_id in image_ids:
            image_info = coco_dataset.loadImgs(image_id)[0]
            image_path = osp.join(image_root_path, image_info["file_name"])
            height, width = image_info['height'], image_info['width']
            print(image_info['file_name'])
            # copy img
            shutil.copy(image_path, osp.join(output_path, "images", flag, image_info["file_name"]))
            label_txt_path = osp.join(output_path, "labels", 
                flag, image_info["file_name"].replace(".jpg", ".txt"))
            # conver label
             # coco anno is x,y,w,h, we need x1,y1,x2,y2
            anno_ids = coco_dataset.getAnnIds(image_id, iscrowd=False)
            with open(label_txt_path, "w") as f:
                if len(anno_ids) > 0:
                    coco_annos = coco_dataset.loadAnns(anno_ids)
                    for _, anno in enumerate(coco_annos):
                        # some annotations have basically no width / height, skip them
                        if anno['bbox'][2] < 1 or anno["bbox"][3] < 1:
                            continue
                        box = np.array(anno["bbox"], dtype=np.float64)
                        box[:2] += box[2:] / 2
                        box[[0, 2]] /= width # normalize
                        box[[1, 3]] /= height

                        if (box[2] > 0.) and (box[3] > 0.):
                            f.write('%g %.6f %.6f %.6f %.6f\n' % (coco80[anno["category_id"]-1], *box))
                f.close()
