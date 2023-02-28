import torch
import pickle

import sys
sys.path.append(r'/home/liangly/my_projects/myYolo/codes')
from models.yolov5_model import Yolov5Model


def convert_yolov5s_weights():
    file = open(r'/home/liangly/my_projects/myYolo/weights/yolov5s_weight.pkl', 'rb')
    ori_model_dict = pickle.load(file)
    # print(ori_model_dict)
    with open('ori_model.txt', 'w') as f:
        for key, value in ori_model_dict.items():
           f.write(key + ":" + str(value.shape) + '\n')
        
def tmp():
    model = Yolov5Model()
    model_state_dict = model.state_dict()
    with open('model.txt', 'w') as f:
        for key, value in model_state_dict.items():
           f.write(key + ":" + str(value.shape) + '\n')


if __name__ == '__main__':
    convert_yolov5s_weights()
    tmp()
