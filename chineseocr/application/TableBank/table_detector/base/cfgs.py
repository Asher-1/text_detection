#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chineseocr
FILE_NAME    :  cfgs
AUTHOR       :  DAHAI LU
TIME         :  2019/7/25 上午10:27
PRODUCT_NAME :  PyCharm
"""
import os

# system path configuration
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
WORK_PATH = os.path.join(ROOT_PATH, "table_detector/base")

OUT_PATH = os.path.join(ROOT_PATH, "tmp")
IMAGE_INFO = os.path.join(ROOT_PATH, "tmp/info.txt")
YAML_FILE = os.path.join(WORK_PATH, "models/config_X101.yaml")
WEIGHTS = os.path.join(WORK_PATH, "models/model_final_res101.pkl")
TEST_PATH = os.path.join(ROOT_PATH, "data/Sampled_Detection_data/Word/images")

# detection configuration
IMAGE_EXT = "jpg"
VIS = False
VERBOSE = True
GPU_MODE = 0
DETECT_THRESHOLD = 0.7


if __name__ == '__main__':
    print("ROOT_PATH = ", ROOT_PATH)
    print("OUT_PATH = ", OUT_PATH)
    print("IMAGE_INFO = ", IMAGE_INFO)
    print("WORK_PATH = ", WORK_PATH)
    print("YAML_FILE = ", YAML_FILE)
    print("WEIGHTS = ", WEIGHTS)
