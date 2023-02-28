import torch
import pickle

import sys
sys.path.append(r'/home/liangly/my_projects/myYolo/codes')
from models.yolov5_model import Yolov5Model


def convert_yolov5s_weights():
    file = open(r'/home/liangly/my_projects/myYolo/weights/yolov5s_weight.pkl', 'rb')
    ori_model_dict = pickle.load(file)
    # print(ori_model_dict)
    # with open('ori_model.txt', 'w') as f:
    #     for key, value in ori_model_dict.items():
    #        f.write(key + ":" + str(value.shape) + '\n')
    ori_model_dict.pop("model.24.anchors")
    model = Yolov5Model()
    model_state_dict = model.state_dict()
    print(len(ori_model_dict))
    print(len(model_state_dict))
    assert len(ori_model_dict) == len(model_state_dict), "字典key无法对齐"
    # with open('model.txt', 'w') as f:
    #     for key, value in model_state_dict.items():
    #        f.write(key + ":" + str(value.shape) + '\n')
    model_weight = {}
    for src_key, dst_key in zip(ori_model_dict.keys(), model_state_dict.keys()):
        model_weight[dst_key] = ori_model_dict[src_key]
    model.load_state_dict(model_weight)
    torch.save(model_weight, 'yolov5_convert.pth')

if __name__ == '__main__':
    convert_yolov5s_weights()
    # tmp()
