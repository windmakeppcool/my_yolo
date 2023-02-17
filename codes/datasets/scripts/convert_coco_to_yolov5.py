import json
import glob
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm


def make_folders(path='../out/'):
    # Create folders
 
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    os.makedirs(path + os.sep + 'images')  # make new labels folder
    return path


def convert_coco_json(json_dir='../coco/annotations/', output='out/'):
    dir = make_folders(path=output)  # output directory
    jsons = glob.glob(json_dir + '*.json')
    coco80 = coco91_to_coco80_class()
 
    # Import json
    for json_file in sorted(jsons):
        fn = output+'/labels/%s/' % Path(json_file).stem.replace('instances_', '')  # folder name
        os.mkdir(fn)
        with open(json_file) as f:
            data = json.load(f)
 
        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
 
        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            if x['iscrowd']:
                continue
 
            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']
 
            # The Labelbox bounding box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
 
            if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                with open(fn + Path(f).stem + '.txt', 'a') as file:
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (coco80[x['category_id'] - 1], *box))


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
 
convert_coco_json("/home/liangly/datasets/coco/annotations/", "/home/liangly/datasets/coco/out")