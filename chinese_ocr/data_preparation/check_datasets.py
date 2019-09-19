#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  check_datasets
AUTHOR       :  DAHAI LU
TIME         :  2019/8/29 下午1:59
PRODUCT_NAME :  PyCharm
"""

from os.path import join
from PIL.Image import open as pit_open
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
    _imagefile = (i for i, j in image_label.items())
    invalid_num = 0
    for file_name in tqdm(list(_imagefile)):
        try:
            img = pit_open(join(image_path, file_name))
            img.verify()
        except Exception as e:
            print("\n cannot open file: {}".format(file_name))
            invalid_num += 1
            continue
    print("detect invalid image number: {}".format(invalid_num))


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/"
    image_path = ROOT_PATH + "syn_images2"
    train_file = ROOT_PATH + "syn_train2.txt"
    test_file = ROOT_PATH + "syn_test2.txt"

    print("check training data...")
    check_invalidation(train_file)

    print("check testing data...")
    check_invalidation(test_file)
