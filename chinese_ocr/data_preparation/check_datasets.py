#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  check_datasets
AUTHOR       :  DAHAI LU
TIME         :  2019/8/29 下午1:59
PRODUCT_NAME :  PyCharm
"""

import os
from PIL import Image
from tqdm import tqdm


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic


def check_invalidation(label_file):
    image_label = readfile(label_file)
    _imagefile = [i for i, j in image_label.items()]
    invalid_num = 0
    for file_name in tqdm(_imagefile):
        try:
            img = Image.open(os.path.join(image_path, file_name))
            img.verify()
        except Exception as e:
            print("\n cannot open file: {}".format(file_name))
            invalid_num += 1
            continue
    print("detect invalid image number: {}".format(invalid_num))


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/"
    image_path = ROOT_PATH + "syn_images2"
    train_file = ROOT_PATH + "syn_train.txt"
    test_file = ROOT_PATH + "syn_test.txt"

    print("check training data...")
    check_invalidation(train_file)

    print("check testing data...")
    check_invalidation(test_file)
