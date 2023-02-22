import os
import os.path as osp


def mkdir(path):
    if not osp.exists(path):
        os.mkdir(path)
