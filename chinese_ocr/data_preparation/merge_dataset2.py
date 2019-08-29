#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  merge_dataset
AUTHOR       :  DAHAI LU
TIME         :  2019/8/14 下午4:59
PRODUCT_NAME :  PyCharm
"""

import os
import shutil
import random
from tqdm import tqdm


def merge_dataset(out_file, input_file1, input_file2):
    with open(out_file, "w") as fw, open(input_file1, "r") as fr1, open(input_file2, "r") as fr2:
        data_list1 = fr1.read().splitlines()
        data_list2 = fr2.read().splitlines()
        merge_list = []
        merge_list.extend(data_list1)
        merge_list.extend(data_list2)

        random.shuffle(merge_list)
        for merge_line in tqdm(merge_list):
            text = "{}\n".format(merge_line)
            fw.write(text)

        for data in tqdm(data_list2):
            image_file = data.split()[0]
            image_path = os.path.join(IMAGE_PATH, image_file)
            shutil.copy(image_path, OUT_PATH)


if __name__ == '__main__':
    # ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data/"
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/"

    IMAGE_PATH = ROOT_PATH + "images2"
    TRAIN_DATA2 = ROOT_PATH + "data_train2.txt"
    TEST_DATA2 = ROOT_PATH + "data_test2.txt"

    TRAIN_DATA = ROOT_PATH + "data_train.txt"
    TEST_DATA = ROOT_PATH + "data_test.txt"

    OUT_PATH = ROOT_PATH + "images"
    TRAIN_OUT = ROOT_PATH + "syn_train.txt"
    TEST_OUT = ROOT_PATH + "syn_test.txt"

    print("start merging training dataset")
    merge_dataset(TRAIN_OUT, TRAIN_DATA, TRAIN_DATA2)
    print("start merging testing dataset")
    merge_dataset(TEST_OUT, TEST_DATA, TEST_DATA2)
